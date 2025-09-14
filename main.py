# main.py - Phase 5 (Advanced Chart + GPT-4o-mini Analysis with 11 coins)
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import numpy as np

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 65.0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=50"

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"Markdown"})

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path,"rb") as f:
        data=aiohttp.FormData()
        data.add_field("chat_id",str(TELEGRAM_CHAT_ID))
        data.add_field("caption",caption)
        data.add_field("photo",f,filename="chart.png",content_type="image/png")
        await session.post(url,data=data)

# ---------------- Fetching ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=15) as r:
            if r.status!=200: return None
            return await r.json()
    except: return None

async def fetch_data(session,symbol):
    t=await fetch_json(session,TICKER_URL.format(symbol=symbol))
    c=await fetch_json(session,CANDLE_URL.format(symbol=symbol))
    out={}
    if t:
        out["price"]=float(t.get("lastPrice",0))
        out["volume"]=float(t.get("volume",0))
    if isinstance(c,list):
        out["candles"]=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c]
        out["times"]=[int(x[0])//1000 for x in c]
    return out

# ---------------- Levels / Trendlines ----------------
def levels(candles,lookback=24):
    if not candles: return (None,None,None)
    arr=candles[-lookback:]
    highs=sorted([c[1] for c in arr],reverse=True)
    lows=sorted([c[2] for c in arr])
    k=min(3,len(arr))
    res=sum(highs[:k])/k; sup=sum(lows[:k])/k; mid=(res+sup)/2
    return sup,res,mid

def trendline(points_x,points_y):
    if len(points_x)<2: return None
    coeffs=np.polyfit(points_x,points_y,1)
    return coeffs

def plot_chart(times,candles,sym,levs):
    dates=[datetime.utcfromtimestamp(t) for t in times]
    o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c=[c[3] for c in candles]
    x=date2num(dates)
    fig,ax=plt.subplots(figsize=(7,4),dpi=100)

    # Black & White candlesticks
    for xi,oi,hi,li,ci in zip(x,o,h,l,c):
        col="white" if ci>=oi else "black"
        edge="black"
        ax.vlines(xi,li,hi,color="black",linewidth=0.6)
        ax.add_patch(plt.Rectangle((xi-0.2,min(oi,ci)),0.4,abs(ci-oi),facecolor=col,edgecolor=edge))

    sup,res,mid=levs
    if res: ax.axhline(res,color="orange",linestyle="--",label=f"Res {res:.2f}")
    if sup: ax.axhline(sup,color="purple",linestyle="--",label=f"Sup {sup:.2f}")
    if mid: ax.axhline(mid,color="gray",linestyle=":",label=f"Mid {mid:.2f}")

    # Trendline approx (last 10 closes)
    if len(c)>=5:
        coeffs=trendline(range(len(c[-10:])),c[-10:])
        if coeffs is not None:
            m,b=coeffs
            xx=np.array(range(len(c[-10:])))
            yy=m*xx+b
            ax.plot(x[-10:],yy,color="blue",linestyle="-.",label="Trendline")

    ax.set_title(sym)
    ax.legend(loc="upper left",fontsize="small")
    fig.autofmt_xdate()
    tmp=NamedTemporaryFile(delete=False,suffix=".png")
    fig.savefig(tmp.name,bbox_inches="tight"); plt.close(fig)
    return tmp.name

# ---------------- OpenAI analysis ----------------
async def analyze_openai(market):
    if not client: return None
    parts=[]
    for s,d in market.items():
        parts.append(f"{s}: price={d.get('price')} vol={d.get('volume')}")
        if d.get("candles"):
            last10=d["candles"][-10:]
            parts.append(f"{s} 30m last10:"+",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10]))
    prompt=(
        "You are an advanced crypto analyst.\n"
        "For each symbol analyze candlesticks, chart patterns, support/resistance, trendlines, volume, open interest, long/short ratio.\n"
        "Detect BUY/SELL opportunities only if strong signals appear.\n"
        "Output format per line:\n"
        "SYMBOL - BIAS - REASON - CONF: <NN>%\n"
        "Where BIAS = BUY/SELL/NEUTRAL, CONF = 0-100 confidence.\n"
        "Reason must briefly mention detected patterns/levels."
        "\n\nData:\n"+ "\n".join(parts)
    )
    resp=await asyncio.get_event_loop().run_in_executor(None,
        lambda: client.chat.completions.create(model=OPENAI_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=800,temperature=0.2))
    return resp.choices[0].message.content.strip()

def parse(text):
    out={}
    if not text: return out
    for ln in text.splitlines():
        parts=[p.strip() for p in ln.split(" - ")]
        if len(parts)<3: continue
        sym=parts[0]; bias=parts[1]; reason=parts[2]; conf=None
        if "CONF" in ln.upper():
            try:
                conf=int(ln.upper().split("CONF")[1].replace(":","").replace("%","").strip())
            except: conf=None
        out[sym]={"bias":bias.upper(),"reason":reason,"conf":conf}
    return out

# ---------------- Loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session,f"Bot online — Phase-5 (11 coins, conf≥{SIGNAL_CONF_THRESHOLD}%)")
        while True:
            tasks=[fetch_data(session,s) for s in SYMBOLS]
            res=await asyncio.gather(*tasks)
            market={s:r for s,r in zip(SYMBOLS,res)}
            txt=await analyze_openai(market)
            parsed=parse(txt)

            for s,info in parsed.items():
                conf=info.get("conf",0)
                bias=info.get("bias","NEUTRAL").upper()
                if conf>=SIGNAL_CONF_THRESHOLD and bias in ("BUY","SELL"):
                    d=market.get(s,{}); levs=levels(d.get("candles"))
                    chart=plot_chart(d.get("times"),d.get("candles"),s,levs)
                    caption=f"🚨 {s} → {bias}\nReason: {info['reason']}\nConf: {conf}%"
                    await send_photo(session,caption,chart)

            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    asyncio.run(loop())

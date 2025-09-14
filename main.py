# main.py - Phase 4.4 (Charts + Entry/SL/TP)
import os, asyncio, aiohttp, time
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

load_dotenv()

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT"]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL","gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=50"

async def send_text(session,text):
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

async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=15) as r:
            if r.status!=200: return None
            return await r.json()
    except: return None

async def fetch_data(session,symbol):
    t=await fetch_json(session,TICKER_URL.format(symbol=symbol))
    c30=await fetch_json(session,CANDLE_URL.format(symbol=symbol,interval="30m"))
    out={}
    if t:
        out["price"]=float(t.get("lastPrice",0))
        out["volume"]=float(t.get("volume",0))
    if isinstance(c30,list):
        out["candles"]=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c30]
        out["times"]=[int(x[0])//1000 for x in c30]
    return out

def levels(candles,lookback=24):
    if not candles: return (None,None,None)
    arr=candles[-lookback:]
    highs=sorted([c[1] for c in arr],reverse=True)
    lows=sorted([c[2] for c in arr])
    k=min(3,len(arr))
    res=sum(highs[:k])/k; sup=sum(lows[:k])/k; mid=(res+sup)/2
    return sup,res,mid

def plot_chart(times,candles,sym,levs):
    dates=[datetime.utcfromtimestamp(t) for t in times]
    o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c=[c[3] for c in candles]
    x=date2num(dates)
    fig,ax=plt.subplots(figsize=(7,4),dpi=100)
    for xi,oi,hi,li,ci in zip(x,o,h,l,c):
        col="green" if ci>=oi else "red"
        ax.vlines(xi,li,hi,color="black")
        ax.add_patch(plt.Rectangle((xi-0.2,min(oi,ci)),0.4,abs(ci-oi),color=col))
    sup,res,mid=levs
    if res: ax.axhline(res,color="orange",linestyle="--")
    if sup: ax.axhline(sup,color="purple",linestyle="--")
    if mid: ax.axhline(mid,color="gray",linestyle=":")
    ax.set_title(sym)
    fig.autofmt_xdate()
    tmp=NamedTemporaryFile(delete=False,suffix=".png")
    fig.savefig(tmp.name,bbox_inches="tight"); plt.close(fig)
    return tmp.name

def trade_levels(price,levs,bias):
    sup,res,mid=levs
    if not price: return None
    if bias=="BUY":
        sl=sup*0.997 if sup else price*0.99
        risk=price-sl
        return {"entry":price,"sl":sl,"tp1":price+risk,"tp2":price+2*risk}
    if bias=="SELL":
        sl=res*1.003 if res else price*1.01
        risk=sl-price
        return {"entry":price,"sl":sl,"tp1":price-risk,"tp2":price-2*risk}
    return None

async def analyze_openai(market):
    if not client: return None
    parts=[]
    for s,d in market.items():
        parts.append(f"{s}: price={d.get('price')} vol={d.get('volume')}")
        if d.get("candles"):
            last10=d["candles"][-10:]
            parts.append(f"{s} 30m last10:"+",".join([f"[{c[0]},{c[1]},{c[2]},{c[3]}]" for c in last10]))
    prompt="Output lines: SYMBOL - BIAS - TF - REASON\n"+ "\n".join(parts)
    resp=await asyncio.get_event_loop().run_in_executor(None, lambda: client.chat.completions.create(model=OPENAI_MODEL,messages=[{"role":"user","content":prompt}],max_tokens=500,temperature=0.2))
    return resp.choices[0].message.content.strip()

def parse(text):
    out={}
    if not text: return out
    for ln in text.splitlines():
        parts=[p.strip() for p in ln.split(" - ")]
        if len(parts)>=3:
            out[parts[0]]={"bias":parts[1],"tf":parts[2],"reason":parts[3] if len(parts)>3 else ""}
    return out

async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session,"Bot online Phase-4.4")
        while True:
            tasks=[fetch_data(session,s) for s in SYMBOLS]
            res=await asyncio.gather(*tasks)
            market={s:r for s,r in zip(SYMBOLS,res)}
            txt=await analyze_openai(market)
            parsed=parse(txt)
            for s,info in parsed.items():
                d=market.get(s,{}); levs=levels(d.get("candles"))
                tl=trade_levels(d.get("price"),levs,info.get("bias"))
                if tl:
                    caption=f"🚨 {s} {info['bias']} TF:{info['tf']}\nReason:{info['reason']}\nEntry:{tl['entry']:.2f} SL:{tl['sl']:.2f} TP1:{tl['tp1']:.2f} TP2:{tl['tp2']:.2f}"
                    chart=plot_chart(d.get("times"),d.get("candles"),s,levs)
                    await send_photo(session,caption,chart)
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    asyncio.run(loop())

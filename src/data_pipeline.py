import logging
from pathlib import Path
from typing import Tuple
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib

from src.config import CONFIG

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ticker alias map  →  Yahoo Finance symbol
# Covers: Indian (NSE/BSE) + Major Global (US, Europe, Asia) + Indices + ETFs
# ─────────────────────────────────────────────────────────────────────────────
_INDIAN_TICKER_ALIASES: dict[str, str] = {

    # =========================================================================
    # INDIAN INDICES
    # =========================================================================
    "NIFTY":          "^NSEI",
    "NIFTY50":        "^NSEI",
    "SENSEX":         "^BSESN",
    "NIFTYBANK":      "^NSEBANK",
    "NIFTYIT":        "^CNXIT",
    "NIFTYPHARMA":    "^CNXPHARMA",
    "NIFTYAUTO":      "^CNXAUTO",
    "NIFTYMETAL":     "^CNXMETAL",
    "NIFTYREALTY":    "^CNXREALTY",
    "NIFTYFMCG":      "^CNXFMCG",
    "NIFTYENERGY":    "^CNXENERGY",
    "NIFTYINFRA":     "^CNXINFRA",
    "NIFTYMIDCAP":    "^NSEMDCP50",
    "NIFTYSMALLCAP":  "^NSMIDCP",

    # =========================================================================
    # NIFTY 50 CONSTITUENTS  (NSE)
    # =========================================================================
    # Banking & Finance
    "HDFCBANK":       "HDFCBANK.NS",
    "HDFC":           "HDFCBANK.NS",
    "ICICIBANK":      "ICICIBANK.NS",
    "ICICI":          "ICICIBANK.NS",
    "SBI":            "SBIN.NS",
    "SBIN":           "SBIN.NS",
    "KOTAKBANK":      "KOTAKBANK.NS",
    "KOTAK":          "KOTAKBANK.NS",
    "AXISBANK":       "AXISBANK.NS",
    "AXIS":           "AXISBANK.NS",
    "INDUSINDBK":     "INDUSINDBK.NS",
    "BANDHANBNK":     "BANDHANBNK.NS",

    # IT & Technology
    "TCS":            "TCS.NS",
    "INFY":           "INFY.NS",
    "INFOSYS":        "INFY.NS",
    "WIPRO":          "WIPRO.NS",
    "HCLTECH":        "HCLTECH.NS",
    "HCL":            "HCLTECH.NS",
    "TECHM":          "TECHM.NS",
    "TECHMAHINDRA":   "TECHM.NS",
    "LTIM":           "LTIM.NS",
    "MPHASIS":        "MPHASIS.NS",
    "PERSISTENT":     "PERSISTENT.NS",
    "COFORGE":        "COFORGE.NS",
    "LTTS":           "LTTS.NS",
    "KPITTECH":       "KPITTECH.NS",

    # Energy & Oil
    "RELIANCE":       "RELIANCE.NS",
    "RIL":            "RELIANCE.NS",
    "ONGC":           "ONGC.NS",
    "BPCL":           "BPCL.NS",
    "IOC":            "IOC.NS",
    "NTPC":           "NTPC.NS",
    "POWERGRID":      "POWERGRID.NS",
    "ADANIGREEN":     "ADANIGREEN.NS",
    "ADANIPOWER":     "ADANIPOWER.NS",
    "TATAPOWER":      "TATAPOWER.NS",
    "TORNTPOWER":     "TORNTPOWER.NS",
    "CESC":           "CESC.NS",

    # Conglomerates & Adani Group
    "ADANIENT":       "ADANIENT.NS",
    "ADANI":          "ADANIENT.NS",
    "ADANIPORTS":     "ADANIPORTS.NS",
    "ADANITRANS":     "ADANITRANS.NS",
    "ADANIGAS":       "ATGL.NS",
    "ATGL":           "ATGL.NS",
    "AWL":            "AWL.NS",
    "NDTV":           "NDTV.NS",

    # Tata Group
    "TATASTEEL":      "TATASTEEL.NS",
    "TATAMOTORS":     "TATAMOTORS.NS",
    "TATACONSUM":     "TATACONSUM.NS",
    "TATACHEM":       "TATACHEM.NS",
    "TATACOMM":       "TATACOMM.NS",
    "TRENT":          "TRENT.NS",
    "TITAN":          "TITAN.NS",
    "VOLTAS":         "VOLTAS.NS",
    "TATAELXSI":      "TATAELXSI.NS",

    # FMCG & Consumer
    "HUL":            "HINDUNILVR.NS",
    "HINDUNILVR":     "HINDUNILVR.NS",
    "ITC":            "ITC.NS",
    "NESTLEIND":      "NESTLEIND.NS",
    "NESTLE":         "NESTLEIND.NS",
    "BRITANNIA":      "BRITANNIA.NS",
    "DABUR":          "DABUR.NS",
    "GODREJCP":       "GODREJCP.NS",
    "MARICO":         "MARICO.NS",
    "COLPAL":         "COLPAL.NS",
    "PIDILITIND":     "PIDILITIND.NS",
    "EMAMILTD":       "EMAMILTD.NS",
    "BAJAJCON":       "BAJAJCON.NS",
    "JYOTHYLAB":      "JYOTHYLAB.NS",
    "VBL":            "VBL.NS",
    "UBL":            "UBL.NS",
    "MCDOWELL":       "MCDOWELL-N.NS",
    "RADICO":         "RADICO.NS",

    # Automobile
    "MARUTI":         "MARUTI.NS",
    "BAJAJ-AUTO":     "BAJAJ-AUTO.NS",
    "BAJAJAUTO":      "BAJAJ-AUTO.NS",
    "HEROMOTOCO":     "HEROMOTOCO.NS",
    "HEROMOTO":       "HEROMOTOCO.NS",
    "EICHERMOT":      "EICHERMOT.NS",
    "ROYALENFIELD":   "EICHERMOT.NS",
    "M&M":            "M&M.NS",
    "MAHINDRA":       "M&M.NS",
    "TVSMOTOR":       "TVSMOTOR.NS",
    "ASHOKLEY":       "ASHOKLEY.NS",
    "MOTHERSON":      "MOTHERSON.NS",
    "BALKRISIND":     "BALKRISIND.NS",
    "APOLLOTYRE":     "APOLLOTYRE.NS",
    "MRF":            "MRF.NS",
    "CEATLTD":        "CEATLTD.NS",
    "BHARATFORG":     "BHARATFORG.NS",
    "SUNDRMFAST":     "SUNDRMFAST.NS",
    "BOSCHLTD":       "BOSCHLTD.NS",
    "ENDURANCE":      "ENDURANCE.NS",
    "EXIDEIND":       "EXIDEIND.NS",

    # Pharma & Healthcare
    "SUNPHARMA":      "SUNPHARMA.NS",
    "DRREDDY":        "DRREDDY.NS",
    "CIPLA":          "CIPLA.NS",
    "DIVISLAB":       "DIVISLAB.NS",
    "BIOCON":         "BIOCON.NS",
    "LUPIN":          "LUPIN.NS",
    "AUROPHARMA":     "AUROPHARMA.NS",
    "TORNTPHARM":     "TORNTPHARM.NS",
    "ALKEM":          "ALKEM.NS",
    "IPCALAB":        "IPCALAB.NS",
    "ABBOTINDIA":     "ABBOTINDIA.NS",
    "PFIZER":         "PFIZER.NS",
    "GLAXO":          "GLAXO.NS",
    "APOLLOHOSP":     "APOLLOHOSP.NS",
    "FORTIS":         "FORTIS.NS",
    "MAXHEALTH":      "MAXHEALTH.NS",
    "LALPATHLAB":     "LALPATHLAB.NS",
    "METROPOLIS":     "METROPOLIS.NS",
    "SYNGENE":        "SYNGENE.NS",

    # Infrastructure & Construction
    "LT":             "LT.NS",
    "LARSENTOUBRO":   "LT.NS",
    "ULTRACEMCO":     "ULTRACEMCO.NS",
    "ULTRACEM":       "ULTRACEMCO.NS",
    "SHREECEM":       "SHREECEM.NS",
    "AMBUJACEMENT":   "AMBUJACEMENT.NS",
    "ACCCEMENT":      "ACC.NS",
    "ACC":            "ACC.NS",
    "JKCEMENT":       "JKCEMENT.NS",
    "RAMCOCEM":       "RAMCOCEM.NS",
    "HEIDELBERG":     "HEIDELBERGCEMENT.NS",
    "NCC":            "NCC.NS",
    "KNRCON":         "KNRCON.NS",
    "PNC":            "PNCINFRA.NS",
    "PNCINFRA":       "PNCINFRA.NS",
    "HG":             "HGINFRA.NS",
    "HGINFRA":        "HGINFRA.NS",
    "IRB":            "IRB.NS",
    "GMRINFRA":       "GMRINFRA.NS",
    "GRINFRA":        "GRINFRA.NS",

    # Metals & Mining
    "JSWSTEEL":       "JSWSTEEL.NS",
    "JSW":            "JSWSTEEL.NS",
    "HINDALCO":       "HINDALCO.NS",
    "VEDL":           "VEDL.NS",
    "VEDANTA":        "VEDL.NS",
    "COALINDIA":      "COALINDIA.NS",
    "NMDC":           "NMDC.NS",
    "SAIL":           "SAIL.NS",
    "NATIONALUM":     "NATIONALUM.NS",
    "MOIL":           "MOIL.NS",
    "HINDCOPPER":     "HINDCOPPER.NS",
    "APLAPOLLO":      "APLAPOLLO.NS",

    # Telecom & Media
    "BHARTIARTL":     "BHARTIARTL.NS",
    "AIRTEL":         "BHARTIARTL.NS",
    "VODAIDEA":       "IDEA.NS",
    "IDEA":           "IDEA.NS",
    "INDUSTOWER":     "INDUSTOWER.NS",
    "ZOMATO":         "ZOMATO.NS",
    "NYKAA":          "FSN.NS",
    "FSN":            "FSN.NS",
    "PAYTM":          "PAYTM.NS",
    "ONE97":          "PAYTM.NS",
    "POLICYBZR":      "POLICYBZR.NS",
    "PB":             "POLICYBZR.NS",
    "DMART":          "DMART.NS",
    "AVENUSUPER":     "DMART.NS",

    # Finance & NBFC
    "BAJFINANCE":     "BAJFINANCE.NS",
    "BAJAJFINSV":     "BAJAJFINSV.NS",
    "SBICARD":        "SBICARD.NS",
    "CHOLAFIN":       "CHOLAFIN.NS",
    "MUTHOOTFIN":     "MUTHOOTFIN.NS",
    "MANAPPURAM":     "MANAPPURAM.NS",
    "LICHSGFIN":      "LICHSGFIN.NS",
    "RECLTD":         "RECLTD.NS",
    "PFC":            "PFC.NS",
    "IRFC":           "IRFC.NS",
    "HUDCO":          "HUDCO.NS",
    "IIFL":           "IIFL.NS",
    "ANGELONE":       "ANGELONE.NS",
    "CDSL":           "CDSL.NS",
    "BSE":            "BSE.NS",
    "MCX":            "MCX.NS",
    "HDFCLIFE":       "HDFCLIFE.NS",
    "SBILIFE":        "SBILIFE.NS",
    "ICICIPRULI":     "ICICIPRULI.NS",
    "LICI":           "LICI.NS",

    # Paint & Chemicals
    "ASIANPAINT":     "ASIANPAINT.NS",
    "BERGERPAINTS":   "BERGERPAINTS.NS",
    "KANSAINER":      "KANSAINER.NS",
    "UPL":            "UPL.NS",
    "SRF":            "SRF.NS",
    "DEEPAKNTR":      "DEEPAKNTR.NS",
    "AAVAS":          "AAVAS.NS",
    "NAVINFLUOR":     "NAVINFLUOR.NS",
    "BALRAMCHIN":     "BALRAMCHIN.NS",

    # Real Estate
    "DLF":            "DLF.NS",
    "OBEROIRLTY":     "OBEROIRLTY.NS",
    "LODHA":          "LODHA.NS",
    "BRIGADE":        "BRIGADE.NS",
    "GODREJPROP":     "GODREJPROP.NS",
    "PRESTIGE":       "PRESTIGE.NS",
    "SOBHA":          "SOBHA.NS",
    "PHOENIXLTD":     "PHOENIXLTD.NS",
    "SUNTECK":        "SUNTECK.NS",

    # Power & Utilities
    "ADANIELEC":      "ADANIENSOL.NS",
    "NHPC":           "NHPC.NS",
    "SJVN":           "SJVN.NS",
    "TORNTPOWER2":    "TORNTPOWER.NS",
    "CESC2":          "CESC.NS",
    "SUZLON":         "SUZLON.NS",
    "INOXWIND":       "INOXWIND.NS",

    # Retail & Consumer Discretionary
    "TITAN2":         "TITAN.NS",
    "MANYAVAR":       "MANYAVAR.NS",
    "VEDANT":         "MANYAVAR.NS",
    "SHOPPERSSTOP":   "SHOPERSTOP.NS",
    "TRENT2":         "TRENT.NS",
    "FRETAIL":        "FRETAIL.NS",
    "VMART":          "VMART.NS",
    "PCJEWELLER":     "PCJEWELLER.NS",
    "KAYNES":         "KAYNES.NS",
    "DIXON":          "DIXON.NS",
    "AMBER":          "AMBER.NS",

    # Logistics & Shipping
    "CONCOR":         "CONCOR.NS",
    "BLUEDART":       "BLUEDART.NS",
    "DELHIVERY":      "DELHIVERY.NS",
    "GATI":           "GATI.NS",
    "TCI":            "TCIL.NS",
    "MAHLOG":         "MAHLOG.NS",

    # Hotels & Hospitality
    "INDHOTEL":       "INDHOTEL.NS",
    "TAJHOTELS":      "INDHOTEL.NS",
    "EIH":            "EIH.NS",
    "LEMONTRE":       "LEMONTREE.NS",
    "LEMONTREE":      "LEMONTREE.NS",
    "CHALET":         "CHALET.NS",

    # Indian ETFs
    "NIFTYBEES":      "NIFTYBEES.NS",
    "JUNIORBEES":     "JUNIORBEES.NS",
    "BANKBEES":       "BANKBEES.NS",
    "GOLDBEES":       "GOLDBEES.NS",
    "LIQUIDBEES":     "LIQUIDBEES.NS",
    "ICICIBANKBEES":  "ICICIBANKBEES.NS",

    # =========================================================================
    # US & GLOBAL TICKERS
    # =========================================================================

    # ── US Indices ────────────────────────────────────────────────────────────
    "SPX":            "^GSPC",
    "SP500":          "^GSPC",
    "S&P500":         "^GSPC",
    "DOW":            "^DJI",
    "DJIA":           "^DJI",
    "NDX":            "^NDX",
    "NASDAQ":         "^IXIC",
    "NASDAQ100":      "^NDX",
    "RUT":            "^RUT",
    "RUSSELL2000":    "^RUT",
    "VIX":            "^VIX",
    "FTSE":           "^FTSE",
    "DAX":            "^GDAXI",
    "CAC40":          "^FCHI",
    "NIKKEI":         "^N225",
    "HANGSENG":       "^HSI",
    "KOSPI":          "^KS11",
    "ASX200":         "^AXJO",
    "SHANGHAI":       "000001.SS",
    "CSI300":         "000300.SS",

    # ── US Mega-Cap Tech (Magnificent 7 + FAANG) ──────────────────────────────
    "APPLE":          "AAPL",
    "AAPL":           "AAPL",
    "MICROSOFT":      "MSFT",
    "MSFT":           "MSFT",
    "GOOGLE":         "GOOGL",
    "GOOGL":          "GOOGL",
    "GOOG":           "GOOG",
    "ALPHABET":       "GOOGL",
    "AMAZON":         "AMZN",
    "AMZN":           "AMZN",
    "META":           "META",
    "FACEBOOK":       "META",
    "NVIDIA":         "NVDA",
    "NVDA":           "NVDA",
    "TESLA":          "TSLA",
    "TSLA":           "TSLA",
    "NETFLIX":        "NFLX",
    "NFLX":           "NFLX",
    "BROADCOM":       "AVGO",
    "AVGO":           "AVGO",

    # ── US Semiconductors ─────────────────────────────────────────────────────
    "INTEL":          "INTC",
    "INTC":           "INTC",
    "AMD":            "AMD",
    "QUALCOMM":       "QCOM",
    "QCOM":           "QCOM",
    "MICRON":         "MU",
    "MU":             "MU",
    "AMDVANCED":      "AMD",
    "TSMC":           "TSM",
    "TSM":            "TSM",
    "ARM":            "ARM",
    "MARVELL":        "MRVL",
    "MRVL":           "MRVL",
    "SKYWORKS":       "SWKS",
    "SWKS":           "SWKS",
    "ONSEMI":         "ON",
    "TXN":            "TXN",
    "TEXASINST":      "TXN",
    "ADI":            "ADI",
    "LAM":            "LRCX",
    "LRCX":           "LRCX",
    "AMAT":           "AMAT",
    "KLAC":           "KLAC",
    "ASML":           "ASML",

    # ── US Financial Services ─────────────────────────────────────────────────
    "JPM":            "JPM",
    "JPMORGAN":       "JPM",
    "BAC":            "BAC",
    "BANKOFAMERICA":  "BAC",
    "GS":             "GS",
    "GOLDMANSACHS":   "GS",
    "MS":             "MS",
    "MORGANSTANLEY":  "MS",
    "WFC":            "WFC",
    "WELLSFARGO":     "WFC",
    "C":              "C",
    "CITIGROUP":      "C",
    "CITI":           "C",
    "BLK":            "BLK",
    "BLACKROCK":      "BLK",
    "V":              "V",
    "VISA":           "V",
    "MA":             "MA",
    "MASTERCARD":     "MA",
    "AXP":            "AXP",
    "AMEX":           "AXP",
    "PYPL":           "PYPL",
    "PAYPAL":         "PYPL",
    "SQ":             "SQ",
    "SQUARE":         "SQ",
    "BLOCK":          "SQ",
    "SCHW":           "SCHW",
    "SCHWAB":         "SCHW",

    # ── US Healthcare & Pharma ────────────────────────────────────────────────
    "JNJ":            "JNJ",
    "JOHNSONANDJOHNSON": "JNJ",
    "UNH":            "UNH",
    "UNITEDHEALTH":   "UNH",
    "PFE":            "PFE",
    "PFIZERUS":       "PFE",
    "ABBV":           "ABBV",
    "ABBVIE":         "ABBV",
    "LLY":            "LLY",
    "LILLY":          "LLY",
    "MRK":            "MRK",
    "MERCK":          "MRK",
    "BMY":            "BMY",
    "BRISTOLMYERS":   "BMY",
    "AMGN":           "AMGN",
    "AMGEN":          "AMGN",
    "GILD":           "GILD",
    "GILEAD":         "GILD",
    "BIIB":           "BIIB",
    "BIOGEN":         "BIIB",
    "MRNA":           "MRNA",
    "MODERNA":        "MRNA",
    "ISRG":           "ISRG",
    "INTUITIVESURGICAL": "ISRG",

    # ── US Consumer & Retail ──────────────────────────────────────────────────
    "WMT":            "WMT",
    "WALMART":        "WMT",
    "COST":           "COST",
    "COSTCO":         "COST",
    "TGT":            "TGT",
    "TARGET":         "TGT",
    "HD":             "HD",
    "HOMEDEPOT":      "HD",
    "LOW":            "LOW",
    "LOWES":          "LOW",
    "MCD":            "MCD",
    "MCDONALDS":      "MCD",
    "SBUX":           "SBUX",
    "STARBUCKS":      "SBUX",
    "NKE":            "NKE",
    "NIKE":           "NKE",
    "PG":             "PG",
    "PROCTERGAMBLE":  "PG",
    "KO":             "KO",
    "COCACOLA":       "KO",
    "PEP":            "PEP",
    "PEPSICO":        "PEP",
    "PM":             "PM",
    "PHILIPMORRIS":   "PM",
    "AMZN2":          "AMZN",

    # ── US Energy ─────────────────────────────────────────────────────────────
    "XOM":            "XOM",
    "EXXON":          "XOM",
    "CVX":            "CVX",
    "CHEVRON":        "CVX",
    "COP":            "COP",
    "CONOCOPHILLIPS": "COP",
    "SLB":            "SLB",
    "SCHLUMBERGER":   "SLB",
    "EOG":            "EOG",
    "MPC":            "MPC",
    "PSX":            "PSX",

    # ── US Industrials & Aerospace ────────────────────────────────────────────
    "BA":             "BA",
    "BOEING":         "BA",
    "CAT":            "CAT",
    "CATERPILLAR":    "CAT",
    "DE":             "DE",
    "DEERE":          "DE",
    "GE":             "GE",
    "HON":            "HON",
    "HONEYWELL":      "HON",
    "LMT":            "LMT",
    "LOCKHEEDMARTIN": "LMT",
    "RTX":            "RTX",
    "RAYTHEON":       "RTX",
    "NOC":            "NOC",
    "UPS":            "UPS",
    "FDX":            "FDX",
    "FEDEX":          "FDX",

    # ── US Cloud & SaaS ───────────────────────────────────────────────────────
    "ORCL":           "ORCL",
    "ORACLE":         "ORCL",
    "CRM":            "CRM",
    "SALESFORCE":     "CRM",
    "SAP":            "SAP",
    "NOW":            "NOW",
    "SERVICENOW":     "NOW",
    "SNOW":           "SNOW",
    "SNOWFLAKE":      "SNOW",
    "PLTR":           "PLTR",
    "PALANTIR":       "PLTR",
    "PANW":           "PANW",
    "PALOALTO":       "PANW",
    "CRWD":           "CRWD",
    "CROWDSTRIKE":    "CRWD",
    "ZS":             "ZS",
    "ZSCALER":        "ZS",
    "DDOG":           "DDOG",
    "DATADOG":        "DDOG",
    "TEAM":           "TEAM",
    "ATLASSIAN":      "TEAM",
    "WDAY":           "WDAY",
    "WORKDAY":        "WDAY",
    "ADBE":           "ADBE",
    "ADOBE":          "ADBE",
    "INTU":           "INTU",
    "INTUIT":         "INTU",
    "UBER":           "UBER",
    "LYFT":           "LYFT",
    "SHOP":           "SHOP",
    "SHOPIFY":        "SHOP",
    "SQ2":            "SQ",
    "COIN":           "COIN",
    "COINBASE":       "COIN",
    "RBLX":           "RBLX",
    "ROBLOX":         "RBLX",
    "SPOT":           "SPOT",
    "SPOTIFY":        "SPOT",
    "PINS":           "PINS",
    "PINTEREST":      "PINS",
    "SNAP":           "SNAP",
    "TWTR":           "X",
    "TWITTER":        "X",
    "ABNB":           "ABNB",
    "AIRBNB":         "ABNB",

    # ── European Stocks ───────────────────────────────────────────────────────
    "SHELL":          "SHEL",
    "SHEL":           "SHEL",
    "NOVO":           "NVO",
    "NOVONORDISK":    "NVO",
    "NVO":            "NVO",
    "LVMH":           "MC.PA",
    "HERMESINTL":     "RMS.PA",
    "NESTLE_CH":      "NESN.SW",
    "ROCHE":          "ROG.SW",
    "NOVARTIS":       "NOVN.SW",
    "TOTALENERGY":    "TTE",
    "TTE":            "TTE",
    "BP":             "BP",
    "BPPLC":          "BP",
    "ASTRAZENECA":    "AZN",
    "AZN":            "AZN",
    "GSK":            "GSK",
    "UNILEVER":       "UL",
    "UL":             "UL",
    "DIAGEO":         "DEO",
    "DEO":            "DEO",
    "RIO":            "RIO",
    "RIOTINTO":       "RIO",
    "VOD":            "VOD",
    "VODAFONE":       "VOD",
    "SIEMENS":        "SIEGY",
    "SIEGY":          "SIEGY",
    "SAP2":           "SAP",
    "VOLKSWAGEN":     "VWAGY",
    "VWAGY":          "VWAGY",
    "BENZ":           "MBGYY",
    "MERCEDES":       "MBGYY",
    "BMW":            "BMWYY",
    "BMWYY":          "BMWYY",
    "BNP":            "BNP.PA",
    "AIRBUS":         "AIR.PA",

    # ── Asian Stocks ──────────────────────────────────────────────────────────
    "SAMSUNG":        "005930.KS",
    "SAMSUNGELEC":    "005930.KS",
    "SK HYNIX":       "000660.KS",
    "SKHYNIX":        "000660.KS",
    "HYUNDAI":        "005380.KS",
    "KIA":            "000270.KS",
    "ALIBABA":        "BABA",
    "BABA":           "BABA",
    "TENCENT":        "0700.HK",
    "MEITUAN":        "3690.HK",
    "XIAOMI":         "1810.HK",
    "JD":             "JD",
    "JDCOM":          "JD",
    "PINDUODUO":      "PDD",
    "PDD":            "PDD",
    "BAIDU":          "BIDU",
    "BIDU":           "BIDU",
    "NETEASE":        "NTES",
    "NTES":           "NTES",
    "NIO":            "NIO",
    "LI":             "LI",
    "XPENG":          "XPEV",
    "XPEV":           "XPEV",
    "SONY":           "SONY",
    "SONYGROUP":      "SONY",
    "TOYOTA":         "TM",
    "TM":             "TM",
    "HONDA":          "HMC",
    "HMC":            "HMC",
    "SOFTBANK":       "SFTBY",
    "SFTBY":          "SFTBY",

    # ── Commodity & Sector ETFs (US-listed) ───────────────────────────────────
    "GLD":            "GLD",
    "GOLD":           "GLD",
    "SLV":            "SLV",
    "SILVER":         "SLV",
    "OIL":            "USO",
    "USO":            "USO",
    "NATURALGAS":     "UNG",
    "UNG":            "UNG",
    "COPPER":         "CPER",
    "WHEAT":          "WEAT",
    "CORN":           "CORN",
    "SOYBEAN":        "SOYB",

    # ── Broad Market ETFs (US-listed) ─────────────────────────────────────────
    "SPY":            "SPY",
    "QQQ":            "QQQ",
    "IWM":            "IWM",
    "DIA":            "DIA",
    "VTI":            "VTI",
    "VOO":            "VOO",
    "VXUS":           "VXUS",
    "EEM":            "EEM",
    "EFA":            "EFA",
    "INDA":           "INDA",
    "FLIN":           "FLIN",
    "VNQ":            "VNQ",
    "XLK":            "XLK",
    "XLF":            "XLF",
    "XLE":            "XLE",
    "XLV":            "XLV",
    "XLI":            "XLI",
    "XLY":            "XLY",
    "XLU":            "XLU",
    "ARKK":           "ARKK",
    "ARKG":           "ARKG",
    "ARKW":           "ARKW",

    # ── Crypto ETFs & Trusts ──────────────────────────────────────────────────
    "IBIT":           "IBIT",
    "FBTC":           "FBTC",
    "GBTC":           "GBTC",
    "ETHA":           "ETHA",
    "BITB":           "BITB",

    # ── Bonds & Fixed Income ETFs ─────────────────────────────────────────────
    "TLT":            "TLT",
    "IEF":            "IEF",
    "SHY":            "SHY",
    "LQD":            "LQD",
    "HYG":            "HYG",
    "BND":            "BND",
    "AGG":            "AGG",
}


def _resolve_ticker(ticker: str) -> str:
    """
    Resolve a ticker to a valid Yahoo Finance symbol.

    Resolution order:
    1. Check the alias map for popular Indian short names.
    2. Try the ticker as-is.
    3. Try <ticker>.NS  (NSE India).
    4. Try <ticker>.BO  (BSE India).

    Returns the first symbol that actually returns data, or the
    original ticker if none succeed (letting the caller raise).
    """
    upper = ticker.upper()

    # 1. Alias map
    if upper in _INDIAN_TICKER_ALIASES:
        resolved = _INDIAN_TICKER_ALIASES[upper]
        logger.info(f"Ticker alias: {ticker!r} → {resolved!r}")
        return resolved

    # 2-4. Probe candidates
    candidates = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
    for candidate in candidates:
        try:
            probe = yf.download(candidate, period="5d", auto_adjust=True, progress=False)
            if not probe.empty:
                if candidate != ticker:
                    logger.info(f"Ticker resolved: {ticker!r} → {candidate!r}")
                return candidate
        except Exception:
            continue

    # Fall back to original — download() will raise a clear error
    logger.warning(f"Could not auto-resolve ticker {ticker!r}; falling back to original symbol")
    return ticker


class DataPipeline:
    """Download, engineer features, scale, and create sequences for LSTM training."""

    def __init__(self):
        self.scaler = None

    def download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If download fails or returns empty data
        """
        # Resolve ticker symbol (handles Indian short names and .NS/.BO suffixes)
        ticker = _resolve_ticker(ticker)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {ticker} data from {start} to {end}")
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

                if df.empty:
                    raise ValueError(f"No data downloaded for {ticker}")

                # Ensure it's a DataFrame and not a Series
                if isinstance(df, pd.Series):
                    df = df.to_frame()

                # Handle MultiIndex columns (when downloading multiple tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                # Lowercase column names for consistency
                df.columns = [str(col).lower() for col in df.columns]

                # Drop NaN rows in OHLCV columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in required_cols if col in df.columns]
                if available_cols:
                    df = df.dropna(subset=available_cols)
                else:
                    raise ValueError(f"Missing OHLCV columns. Got columns: {list(df.columns)}")

                if df.empty:
                    raise ValueError(f"All data for {ticker} was NaN after cleaning")

                logger.info(f"Downloaded {len(df)} rows for {ticker}")
                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
                    raise ValueError(f"Failed to download {ticker}: {e}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using numpy/pandas (no external TA library).

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with added features
        """
        logger.info("Adding technical indicators")
        df = df.copy()

        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Average
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['BBL'] = sma20 - (std20 * 2)
        df['BBU'] = sma20 + (std20 * 2)

        # Drop first 50 rows (NaN from indicator warm-up)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values from indicator calculations")

        return df


    def scale(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Scale data using MinMaxScaler.
        
        Args:
            df: DataFrame with features
            fit: If True, fit scaler on this data. If False, use existing scaler.
            
        Returns:
            Scaled numpy array and scaler object
        """
        if fit:
            logger.info("Fitting MinMaxScaler on training data")
            self.scaler = MinMaxScaler()
            # Select only feature columns (exclude index)
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled = self.scaler.fit_transform(df[feature_cols])
            joblib.dump(self.scaler, CONFIG.SCALER_PATH)
            logger.info(f"Scaler saved to {CONFIG.SCALER_PATH}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call scale(fit=True) first.")
            logger.info("Scaling data with existing scaler")
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled = self.scaler.transform(df[feature_cols])
        
        return scaled, self.scaler

    def create_sequences(self, scaled_data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM.
        
        Args:
            scaled_data: Scaled feature array (n_samples, n_features)
            lookback: Number of timesteps to look back
            
        Returns:
            X: (n_samples, lookback, n_features), y: (n_samples, 1)
        """
        logger.info(f"Creating sequences with lookback={lookback}")
        
        X, y = [], []
        # Close price is first feature (index 0)
        close_col_idx = 0
        
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i + lookback])
            y.append(scaled_data[i + lookback, close_col_idx])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        logger.info(f"Created {len(X)} sequences. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into train/val/test chronologically (no shuffling).
        
        Args:
            X: Feature sequences
            y: Target values
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n = len(X)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Verify no data leakage
        assert train_end < val_end < len(X), "Data leakage detected in split"
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            train_loader, val_loader
        """
        logger.info("Creating PyTorch DataLoaders")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader

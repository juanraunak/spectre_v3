from fastapi import FastAPI
from fetch_data import router as fetch_data_router
from shade_1 import router as shade_router
from mirage_db_2 import router as mirage_router
from cipher_3 import router as cipher_router
from fractal_4 import router as fractal_router
from spectre_spider_5 import router as spider_router
from atlas_6 import router as atlas_router


app = FastAPI(title="Linkedin OnBoarding API", version="1.0.0")

app.include_router(fetch_data_router, prefix="/fetch", tags=["Fetch Data"])
app.include_router(shade_router, prefix="/shade", tags=["Shade"])
app.include_router(mirage_router, prefix="/mirage2", tags=["Mirage"])
app.include_router(cipher_router, prefix="/cipher", tags=["Cipher"])
app.include_router(fractal_router, prefix="/fractal", tags=["Fractal"])
app.include_router(spider_router, prefix="/spider", tags=["Spider"])
app.include_router(atlas_router, prefix="/atlas", tags=["Atlas"])
from pydantic import BaseModel

class WaterQuality(BaseModel):
    ph: str
    Hardness: str
    Solids: str
    Chloramines: str
    Sulfate: str
    Conductivity: str
    Organic_carbon: str
    Trihalomethanes: str
    Turbidity: str
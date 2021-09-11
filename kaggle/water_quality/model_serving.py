import uvicorn
from fastapi import FastAPI
import pickle
from WaterQualityClass import WaterQuality

app = FastAPI()

ph_max, ph_min = 13.999999999999998, 0.2274990502021987
Hardness_max, Hardness_min =  317.33812405558257, 47.432
Solids_max, Solids_min =  56488.67241273919, 320.942611274359
Chloramines_max, Chloramines_min =  13.127000000000002, 1.3908709048851806
Sulfate_max, Sulfate_min =  481.0306423059972, 129.00000000000003
Conductivity_max, Conductivity_min =  753.3426195583046, 201.6197367551575
Organic_carbon_max, Organic_carbon_min =  27.00670661116601, 2.1999999999999886
Trihalomethanes_max, Trihalomethanes_min =  124.0, 8.577012932983806
Turbidity_max, Turbidity_min = 6.494748555990993, 1.45

model = pickle.load(open('water_quality_best_model.pkl', 'rb'))
predictions_classes = {0: "Not Potable", 1: "Potable"}

@app.get('/')
def Hello_Welcome():
    return {"message": "Hello from water quality predictions"}


@app.post('/waters/predict')
def waterQualityPredict(data: WaterQuality):
    data_dict = data.dict()
    ph = (float(data_dict['ph']) - ph_min) / (ph_max - ph_min)
    hardness = (float(data_dict['Hardness']) - Hardness_min) / (Hardness_max - Hardness_min)
    solids = (float(data_dict['Solids']) - Solids_min) / (Solids_max - Solids_min)
    chloramines = (float(data_dict['Chloramines']) - Chloramines_min) / (Chloramines_max - Chloramines_min)
    sulfate = (float(data_dict['Sulfate']) - Sulfate_min) / (Sulfate_max - Sulfate_min)
    conductivity = (float(data_dict['Conductivity']) - Conductivity_min) / (Conductivity_max - Conductivity_min)
    organic_carbon = (float(data_dict['Organic_carbon']) - Organic_carbon_min) / (Organic_carbon_max - Organic_carbon_min)
    trihalomethanes = (float(data_dict['Trihalomethanes']) - Trihalomethanes_min) / (Trihalomethanes_max - Trihalomethanes_min)
    turbidity = (float(data_dict['Turbidity']) - Turbidity_min) / (Turbidity_max - Turbidity_min)

    inf_features = [[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]]
    results = model.predict(inf_features)
    results = results.tolist()[0]

    resp = {"Predicted quality": predictions_classes[results]}

    return resp


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9999)
import gradio as gr
import pickle

model=pickle.load(open("../model/model.pkl", "rb"))

def predict_Hours(price):
    return model.predict([[price]])[0]/100
    

demo= gr.Interface(fn=predict_Hours,inputs= gr.Number(label="price"), outputs=gr.Number(label=["square_feet", "bedrooms", "location_score"]))


if __name__=="__main__":
    demo.launch(share=True)
    

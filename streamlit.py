import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

def load_model():
    model = tf.keras.models.load_model('intermediate_amcc.keras')
    return model

def preprocessing_image(image):
    target_size=(64,64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict(model, image) :
    return model.predict(image, batch_size=1)

def interpret_prediction(prediction):
    if prediction.shape[-1] == 1 :
        score = prediction[0][0]
        predicted_class = 0 if score <= 0.5 else 1 
        confidence_scores = [score, 1-score, 0]
    else :
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
    return predicted_class,confidence_scores

def main():
    st.set_page_config(
    page_title="Klasifikasi Kucing & Anjing",
    page_icon="🐾",
    layout="wide"
    )

    with st.sidebar:
        st.markdown("### ℹ️ Tentang Aplikasi")
        st.write("""
        Aplikasi ini dibuat oleh Abyan hisyam untuk mengidentifikasikan hewan anjing atau kucing.
    
        Cara penggunaan:
        1. Upload gambar
        2. Klik tombol 'Analisis'
        3. Lihat hasil prediksi
        
        Jika Anda mengalami masalah:
        1. Pastikan format gambar sesuai (JPG/JPEG/PNG)
        2. Gambar menunjukkan kucing atau anjing dengan jelas
        3. Pastikan gambar memiliki resolusi yang baik
    
        Untuk hasil terbaik:
        - Gunakan gambar dengan pencahayaan yang baik
        - Pastikan hewan terlihat jelas dalam frame
        - Hindari gambar yang terlalu blur atau gelap
    

        """)



    st.title("🐱 Tempat Cek Gambar Kucing & Anjing 🐶")

    try: 
        model =load_model()
        # st.sidebar("Model output shape :", model.output_shape)
    except Exception as err:
        st.error(f"error : {str(err)}")
        return
    
    uploader = st.file_uploader("Pilih gambar.....", type=['jpg','jpeg', 'png'])

    if uploader is not None:
        try:
                col1, col2 = st.columns([2,1])
                with col1:
                    image = Image.open(uploader)
                    st.image(image, caption="ini gambarmu", use_column_width=True)
                
                with col2:
                    if st.button('Cek Gambar', use_container_width=True):
                        with st.spinner('Baru Loading bentar'):
                            processed_image = preprocessing_image(image)
                            prediction = predict(model, processed_image)
                            predicted_class,confidence_scores = interpret_prediction(prediction)
                            class_name = ['Kucing🐱','Anjing🐶']
                            result = class_name[predicted_class]
                            st.success(f"Hasil prediksinya adalah : {result.capitalize()}")
                            progress_value = int(confidence_scores[predicted_class] * 100)
                            progress_bar = st.progress(progress_value)
                            progress_bar.progress(progress_value)

                            st.write(f"Tingkat kemiripan: {progress_value}%")
                            
                            

        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("Pilih file yang benr dong")
            st.write(f"nih errornya : {(err)}")


if __name__ == "__main__":
    main()
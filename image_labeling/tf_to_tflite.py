import tensorflow as tf

# TensorFlow 모델 로드
model = tf.keras.models.load_model('./image_labeling/model/best_mobile_net.h5')

# TensorFlow Lite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TensorFlow Lite 모델 저장
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite format and saved as 'model.tflite'.")
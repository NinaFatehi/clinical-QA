import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("./models/clinical_summary")
#tokenizer = T5Tokenizer.from_pretrained("./models/clinical_summary")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# Streamlit app layout
st.title("Clinical Question Answering")

# Input fields for question and context
question = st.text_input("Enter your clinical question:")
context = st.text_area("Enter clinical note (context) here:", height=300)

# Generate answer when the button is clicked
if st.button("Get Answer"):
    # Prepare the input text by concatenating question and context
    input_text = f"question: {question}  context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate answer
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
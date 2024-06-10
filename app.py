from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForCausalLM
import torch 
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("Vangmayy/Bollywood-Summary-Generator")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_inference(input_text):
    input_text = str(input_text)
    input_ids = tokenizer.encode(input_text, return_tensors = "pt")
    input_ids.to(device)
    new_model = AutoModelForCausalLM.from_pretrained("Vangmayy/Bollywood-Summary-Generator")
    output = new_model.generate(input_ids, max_length = 5000, num_return_sequences = 1)
    output = tokenizer.decode(output[0], skip_special_tokens = True)
    return output

genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]

with gr.Blocks() as intf:
    gr.Markdown("## Movie Summary Generator")
    
    with gr.Row():
        genre_checkboxes = gr.CheckboxGroup(choices=genres, label="Select Genres")
        summary_output = gr.Textbox(label="Generated Summary")

    generate_button = gr.Button("Generate Summary")
    
    def on_click(selected_genres):
        return run_inference(selected_genres)

    generate_button.click(on_click, inputs=genre_checkboxes, outputs=summary_output)

intf.launch()

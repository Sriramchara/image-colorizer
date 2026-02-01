import gradio as gr
import glob
import os
import subprocess
import sys
import cv2
from inference import load_model, colorize_image

# Global variables
current_model = None
current_ckpt_path = None
process = None  # Training process

def get_checkpoints():
    ckpts = glob.glob("checkpoints/*.pth")
    # Sort by modification time (newest first)
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts

def run_inference(img, ckpt_path):
    global current_model, current_ckpt_path
    
    if img is None:
        return None
        
    if not ckpt_path:
        raise gr.Error("Please select a checkpoint first. (Run Training if none exist!)")
        
    # Load model if changed
    if current_model is None or current_ckpt_path != ckpt_path:
        try:
            print(f"Loading model from {ckpt_path}...")
            current_model = load_model(ckpt_path)
            current_ckpt_path = ckpt_path
        except Exception as e:
            raise gr.Error(f"Failed to load model: {str(e)}")
            
    # Run inference
    try:
        # img is RGB from Gradio, convert to BGR for opencv logic in inference.py
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res_bgr = colorize_image(img_bgr, current_model)
        # Convert back to RGB for Gradio
        res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
        return res_rgb
    except Exception as e:
        raise gr.Error(f"Inference failed: {str(e)}")

def start_training(epochs, batch_size, size, dataset_mode):
    global process
    if process is not None and process.poll() is None:
        return "Training is already running!"
        
    cmd = [
        sys.executable, "src/train.py",
        "--epochs", str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--size", str(int(size))
    ]
    
    if dataset_mode == "Fast (1000 images)":
        cmd.extend(["--limit", "1000"])
    elif dataset_mode == "Balanced (40,000 images)":
        cmd.extend(["--limit", "40000"])
    else:
        # Full Scale - No limit
        pass
    
    try:
        # Start detached process
        process = subprocess.Popen(cmd, cwd=os.getcwd())
        return f"Training started! (PID: {process.pid})\nCheck the terminal for progress bars.\nCheckpoints will appear in 'checkpoints/' folder."
    except Exception as e:
        return f"Failed to start training: {str(e)}"

def refresh_checkpoints():
    return gr.update(choices=get_checkpoints())

with gr.Blocks(title="Image Colorizer") as demo:
    gr.Markdown("# Image Colorizer with U-Net & Perceptual Loss")
    
    with gr.Tabs():
        # --- TAB 1: INFERENCE ---
        with gr.Tab("Colorize"):
            with gr.Row():
                with gr.Column():
                    ckpt_dropdown = gr.Dropdown(label="Select Checkpoint", choices=get_checkpoints(), interactive=True)
                    refresh_btn = gr.Button("Refresh Checkpoints", size="sm")
                    input_img = gr.Image(label="Grayscale Input", type="numpy")
                    run_btn = gr.Button("Colorize", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="Colorized Output")
            
            refresh_btn.click(refresh_checkpoints, outputs=ckpt_dropdown)
            run_btn.click(run_inference, inputs=[input_img, ckpt_dropdown], outputs=output_img)

        # --- TAB 2: TRAINING ---
        with gr.Tab("Train"):
            gr.Markdown("### Train a New Model")
            gr.Markdown("_Ensure you have downloaded data using `python src/download_data.py` first!_")
            
            with gr.Row():
                epochs_input = gr.Number(label="Epochs", value=20, precision=0)
                batch_input = gr.Number(label="Batch Size", value=4, precision=0)
                size_input = gr.Number(label="Image Size", value=128, precision=0)
            
            with gr.Row():
                dataset_mode = gr.Dropdown(
                    label="Dataset Mode", 
                    choices=["Fast (1000 images)", "Balanced (40,000 images)", "Full Scale (All images)"],
                    value="Balanced (40,000 images)",
                    interactive=True
                )

            train_btn = gr.Button("Start Training", variant="stop")
            train_status = gr.Textbox(label="Status", interactive=False)
            
            train_btn.click(start_training, inputs=[epochs_input, batch_input, size_input, dataset_mode], outputs=train_status)

if __name__ == "__main__":
    demo.launch()

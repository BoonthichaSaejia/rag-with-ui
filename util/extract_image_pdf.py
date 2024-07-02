import fitz  # PyMuPDF
import io
from PIL import Image
import os

def extract(pdf_path, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Iterate through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)
        
        # Print the number of images found on this page
        print(f"[INFO] Found {len(image_list)} images on page {page_number + 1}")
        
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            source_file = "".join(pdf_path.split('/')[-1].split(".")[:-1])            
            # Save the image
            image = Image.open(io.BytesIO(image_bytes))
            image_path = f"{output_folder}/doc{source_file}_page{page_number + 1}_image{image_index + 1}.{image_ext}"
            image.save(image_path)
            print(f"[INFO] Saved image at {image_path}")

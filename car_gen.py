from transformers import YolosFeatureExtractor, YolosForObjectDetection
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI ,UploadFile, File, HTTPException
from github import Github
from rembg import remove
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import random
import json
import uuid



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Github access token
g = Github('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')

#getting the repository to upload images
repo = g.get_repo('763t7812/car_background')

# def mask_plate(image):
    
#   url = './images/original/car6.jpg'
#   image = Image.open(url)
#   feature_extractor = YolosFeatureExtractor.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')
#   model = YolosForObjectDetection.from_pretrained('nickmuchi/yolos-small-rego-plates-detection')
#   inputs = feature_extractor(images=image, return_tensors="pt")
#   outputs = model(**inputs)
# #   print(outputs)
#   # model predicts bounding boxes and corresponding face mask detection classes
#   logits = outputs.logits
#   bboxes = outputs.pred_boxes
# #   print("boxes: ",bboxes)
#   # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
#   target_sizes = torch.tensor([image.size[::-1]])
#   results = feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
#       0
#   ]
#   print(results)
#   for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     if label == 1:
#         print(
#             f"Detected {model.config.id2label[label.item()]} with confidence "
#             f"{round(score.item(), 3)} at location {box}"
#         )
#         xmin = round(box[0].item())
#         ymin = round(box[1].item())
#         xmax = round(box[2].item())
#         ymax = round(box[3].item())


#   print((xmin, ymin, xmax, ymax))
#   cropped_part = image.crop((xmin, ymin, xmax, ymax))

#       # Apply a blur filter to the cropped part
#   blurred_part = cropped_part.filter(ImageFilter.GaussianBlur(20))

#       # Paste the blurred part back to the image
#   image.paste(blurred_part, (xmin, ymin))

#       # Save the modified image
#   blurred_image_path = "blurred_image.png"
#   image.save(blurred_image_path)
#   return image


def upload_images(car):
    #generating unique ID
    id = uuid.uuid4()

    # Load your image 
    #img_url = './images/original/car6.jpg'
    #img_name = img_url.split('/')[-1]
    #git_img = open(img_url,'rb').read()
    git_img = car.file.read()
    print("input : ",type(git_img))
    repo.create_file(f'original/car{id}.png', 'car image', git_img, branch='main')

    output_path = f'./images/masked/car{id}.png'
    with open(output_path, 'wb') as f:
        subject = remove(git_img, only_mask = True,)
        print("masked: ",type(subject))
        mask = Image.open(BytesIO(subject))
        mask = np.array(mask)
        inverted_mask = np.bitwise_not(mask)
        print("inverted mask: ",type(inverted_mask))
        mask = Image.fromarray(inverted_mask)
        print(type(mask))
        print("Test: ",type(mask))
        img_byte_arr = BytesIO()   #Create a BytesIO object
        mask.save(img_byte_arr, format='PNG')   # Save the image to the BytesIO object, in a specific format
        git_mask = img_byte_arr.getvalue()  # Retrieve the byte data
        repo.create_file(f'masked/car{id}.png', 'masked image', git_mask, branch='main')
        print(type(git_mask))
        f.write(git_mask)
    return id

def prompt_generate():

    prompts = [
    "A sleek car gleams under the city lights, with towering skyscrapers in the background. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing vibrant urban energy.",
    "A stylish car cruises through a winding forest road, surrounded by lush greenery. Shot in 8K with a Canon EOS R3 and Nikon lens, showcasing natural beauty.",
    "A classy car sits in a quaint cobblestone alley, with historic buildings as the backdrop. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing timeless charm.",
    "A modern car speeds along a desert highway, with vast sand dunes stretching into the distance. Shot in 8K with a Canon EOS R3 and Nikon lens, highlighting the desert landscape.",
    "A luxurious car drives through a tunnel of autumn trees, with colorful foliage overhead. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing the beauty of fall.",
    "A sporty car zips through a bustling city intersection, surrounded by busy streets. Shot in 8K with a Canon EOS R3 and Nikon lens, showcasing urban life.",
    "A vintage car rests in front of an old lighthouse, with crashing waves in the background. Shot in 8K with a Canon EOS R3 and Nikon lens, evoking a sense of nostalgia.",
    "A convertible car cruises along a coastal road, with the sun setting over the ocean. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing the magic of a beach sunset.",
    "A futuristic car glides through a sleek cityscape, with neon lights illuminating the skyline. Shot in 8K with a Canon EOS R3 and Nikon lens, showcasing high-tech elegance.",
    "A rugged car conquers a rocky mountain trail, with breathtaking vistas in the distance. Shot in 8K with a Canon EOS R3 and Nikon lens, embracing the spirit of adventure.",
    "A compact car weaves through a charming European village, with colorful buildings lining the streets. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing European charm.",
    "A family car drives through a serene countryside, with rolling hills and farm fields in the background. Shot in 8K with a Canon EOS R3 and Nikon lens, offering a peaceful scene.",
    "A high-performance car races along a winding racetrack, with cheering crowds in the grandstands. Shot in 8K with a Canon EOS R3 and Nikon lens, capturing the excitement of motorsport.",
    "A luxury car parks in front of a grand mansion, with manicured gardens as the backdrop. Shot in 8K with a Canon EOS R3 and Nikon lens, exuding opulence.",
    "A rugged off-road vehicle traverses a rugged mountain terrain, with snow-capped peaks in the distance. Shot in 8K with a Canon EOS R3 and Nikon lens, embracing the call of the wild."
]

  # Generate a random integer from 0 to 29
    num = random.randint(0, 14)
    prompt = prompts[num]
    print(num, "=> prompt: ", prompt)
    return prompt

def stable_diffusion(id,prompt):
    url = "https://modelslab.com/api/v6/image_editing/inpaint"
    try:
        payload = json.dumps({
        "key": "eOzJnG5mSqUwrsPE074gIpjKjOPwAcQBU6hQIj8YHb1VmwAID8mvDIJn62jz",
        "prompt":prompt,
        "negative_prompt":"(((unrealistic proportions))), Painting, Drawing, cartoon, sketch,blurry, bad lighting, disfigured, poorly drawn details, mutation, mutated, (extra parts), (ugly), (poorly designed wheels), fused wheels, messy drawing, broken windows censor, censored, censor_bar, multiple headlights, (mutated shape and form:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad lighting, liquid metal, disfigured, malformed, mutated, design nonsense, text font ui, error, malformed doors, long chassis, blurred, lowers, low res, bad proportions, bad shadow, uncoordinated design, unnatural body, fused windows, bad windows, huge spoilers, poorly drawn spoilers, extra spoilers, liquid body, heavy body, missing parts, huge bumpers, huge wheels, bad wheels, fused wheel, missing wheel, disappearing doors, disappearing hood, disappearing trunk, disappearing wheels, fused lights, bad lights, poorly drawn lights, extra lights, liquid lights, heavy lights, missing lights, old photo, low res, black and white, black and white filter, colorless, unrealistic background, artificial scenery, mismatched environment, floating car, unanchored objects, disconnected shadows, implausible reflections, cluttered background, overly simplistic background, distracting elements",
        "init_image": f"https://github.com/763t7812/car_background/blob/main/original/car{id}.png?raw=true",
        "mask_image": f"https://github.com/763t7812/car_background/blob/main/masked/car{id}.png?raw=true",
        "width": "1024",
        "height": "1024",
        "samples": "1",
        "num_inference_steps": "40",
        "safety_checker": False,
        "enhance_prompt": "yes",
        "guidance_scale": 2.5,
        "strength": 0.7,
        "seed": None,
        "webhook": None,
        "track_id": None
        })

        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response)
        print(response.text)
        res =json.loads(response.text)
        # res2 = res['proxy_links'][0]
        # url2 = res2.replace("\/","/")
        # print("url2: ",url2)
        res = res['output'][0]
        url = res.replace("\/","/")
        edited_img =  Image.open(BytesIO(requests.get(url).content))
        edited_img.save(f'./images/edited/car{id}.png')
        return url
    except Exception as e:
        print("Error: ",e)
        return "Error: ",e


def delete_git_images(id):
    #Clean up images from GitHub
    mask = repo.get_contents(f"masked/car{id}.png", ref="main")
    repo.delete_file(mask.path, "remove masked image", mask.sha, branch='main')
    original = repo.get_contents(f"original/car{id}.png", ref="main")
    repo.delete_file(original.path, "remove cloth image", original.sha, branch='main')


@app.post("/")
async def car_background(car: UploadFile = File(...)):
    id = upload_images(car)
    prompt = prompt_generate()
    output = stable_diffusion(id, prompt)
    print(output)
    delete_git_images(id)
    return output
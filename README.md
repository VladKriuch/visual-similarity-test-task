# Visual Product Search

## Setup

1. Define `DATA` variable in `.env` file in project root (There's an example for this in .env file)
2. To set up and run the project, use Docker from project root dir:

```sh
docker compose build
docker compose up
```

!ATTENTION: First run may be long ( up to 1-2 hours ), because docker will parse a dataset to create db!

## Technical Approach

This project implements a visual product search system using a combination of object detection, OCR, and image embeddings:

### Dataset preparation
For dataset preparation `imagededup` library was used in order to avoid duplicates in images

### Pipeline
Each image is processed with 4 steps
1. **Object Detection**: The pipeline utilizes YOLO trained on the Open Images V7 dataset to detect products in an image. Custom reclassifying is done afterwards in order to narrow down the possible classes. Thus, different rois proposed for different objects in images. It increases the amount of vectors in dataset from 8k (amount of images) to ~30k
1.5 **Additional post processing** - Some of YOLO classes overlap over each other, thus, additional non-max suppression is done after category generalization. Also, for shoes, there's algorithm to combine close 2 shoes into pair. However, that it is not successfull in every case.
2. **Generating Embeddings**: Generates embeddings using OpenAI's CLIP model to compare images/text and find visually similar products. CLIP performs very well in this task. For general images(not detected by YOLO), CLIP is also used to obtain the category for the image.
The embeddings are generated for each image and each roi in the image. 
3. **Elasticsearch**: Stores embeddings in the image_vector fields - `image_vector`.

### Input image pipeline
When user inputs some image, the streamlit app automatically preprocess it by detecting regions of interest using YOLO model. They are displayed as a red dots on the image, so user can just click on them and obtained similarity search results.

User can also choose to draw custom bounding box. The category in this case is decided by using CLIP prediction.

After user selected the region -> The clip model is used in order to obtain embeddings and then the elasticsearch performs filtering by category and index search.

## Project Architecture
| app - Main app folder

| | src - source files

| | | models - folder with models for detection(yolo), embedding(clip), ocr(easyocr) # OCR is NOT used, due to being slow, however the functionality for usage is ready and it can be connected quite quickly

| | | elastic_db.py - abstraction for communication with elastic database

| | static - static files used for re-categorization

| | app.py - streamlit app (mainly the streamlit stuff is here. all the logic is encapsulated in pipeline.py)

| | pipeline.py - abstraction for using models and communicating with elastic db

| | populate_db.py - script used for populating db with images from datatset

| | requirements.txt - reqs 

| | Dockerfile

| docker-compose.yml

| README.md

## Libraries Used

The project relies on the following libraries:

- `imagededup==0.3.1` - Library for near-duplicate image detection in dataset
- `regex` - needed by clip
- `ftfy` - needed by clip
- `git+https://github.com/openai/CLIP.git` - CLIP model
- `elasticsearch==8.17.1` - elasticsearch 
- `opencv-python-headless~=4.11.0.86` - headless opencv version
- `ultralytics~=8.3.70` - ultralytics for YOLO
- `python-dotenv`
- `tqdm`
- `streamlit==1.38.0`
- `streamlit-drawable-canvas` - for drawing and bounding box selection
- `easyocr` - OCR lib


## Possible improvements

Just some things here that I would like to try but haven't had a chance to.

1. Logo detection
2. Compare faiss to elasticdb. I thought that if the system must be scalable, then we would likely use elasticdb. However, we could also use elasticdb for filtering some subset and then give it to faiss. 
3. NSFW detection. (There's couple of nsfw images in data, the open-source libs worked poorly for this, need more research time on this)
4. Segmentation after detection. It would add more time to pipeline but the results may be worth it.
5. Some kind of celebrity detection.
6. Fine-tuning detection model on chosen classes.
7. Dominating color and pattern recognition.


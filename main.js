//Handle Image Input
function handleImageInput(event){
    const fileInput = event.target;
    const file = fileInput.files[0];
    if (file){
        const reader = new FileReader();
        reader.onload = function (e) {
            const imgMain = document.getElementById("canva-img");
            imgMain.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
}


//Compute Color for Labels
function computeColorforLabels(className){
    if(className=='person'){
        color=[85, 45, 255,200];
      }
      else if (className='cup'){
        color=[255, 111, 0, 200]
      }
      else if (className='cellphone'){
        color=[200, 204, 255, 200]
      }
      else{
        color=[0,255,0,200];
      }
      return color;
}

function computeColorForLabel_s(classID){
    if(classID == 0){
        color=[85, 45, 255, 255]
    }
    else if(classID == 2){
        color=[222, 82, 175, 255]
    }
    else if(classID == 3){
        color=[0, 204, 255, 255]
    }
    else if(classID == 4){
        color = [0, 149, 255, 255]
    }
    else{
        color = [255,111,111,255]
    }
    return color;
}




function drawBoundingBox(predictions, image){
    predictions.forEach(
        prediction => {
            const bbox = prediction.bbox;
            const x = bbox[0];
            const y = bbox[1];
            const width = bbox[2];
            const height = bbox[3];
            const className = prediction.class;
            const confScore = prediction.score;
            const color = computeColorforLabels(className)
            console.log(x, y, width, height, className, confScore);
            let point1 = new cv.Point(x, y);
            let point2 = new cv.Point(x+width, y+height);
            cv.rectangle(image, point1, point2, color, 2);
            const text = `${className} - ${Math.round(confScore*100)/100}`;
            const font =  cv.FONT_HERSHEY_TRIPLEX;
            const fontsize = 0.70;
            const thickness = 1;
            //Get the size of the text
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            const textMetrics = context.measureText(text);
            const twidth = textMetrics.width;
            console.log("Text Width", twidth);
            cv.rectangle(image, new cv.Point(x, y-20), new cv.Point(x + twidth + 150,y), color, -1);
            cv.putText(image, text, new cv.Point(x, y-5), font, fontsize, new cv.Scalar(255, 255, 255, 255), thickness);
        }
    )
}


function downloadImage() {
    //Get the Canvas Element
    const canvas = document.getElementById('main-canva');

    //Create an Anchor Element to Trigger the Download
    const link = document.createElement('a');

    //Set the download attribute with a filename 
    link.download = 'color_detection.png';

    //Convert the Canvas Content to a data URL
    const dataURL = canvas.toDataURL();

    //Set the href attribute of the anchor with the Data URL
    link.href = dataURL;

    //Append the anchor to the document
    document.body.appendChild(link);

    //Trigger a Click on the anchor to start the download
    link.click();

    //Remove the anchor element from the document
    document.body.removeChild(link);
}



function openCvReady(){
    // this for write cv code
    cv["onRuntimeInitialized"]=()=>{

        console.log("OpenCV Ready")

        //read an image from the image source and convert to opencv format
        let imgMain = cv.imread("canva-img");//reading
        //let variable 
        cv.imshow("main-canva",imgMain); //displaying image inside the canvas
        //we need to delete the image to free the memmory   
        imgMain.delete();



        // ********* Image handling ***********
        document.getElementById("image_up").addEventListener("change",handleImageInput);





        //-------------- Now we want the html code in the js -------------
        //for the we using document
        //we are taking that getElementById("button_rgb") 
        //once it clicked we are operating the function


       /**** for rgb ***/
        document.getElementById("button_rgb").onclick = function(){

            let imgMain = cv.imread("canva-img");//reading
            cv.imshow("main-canva",imgMain); //displaying at the main_canva
            imgMain.delete();//we need to delete the image to free the memmory

        };


        /**** for Object Detection Image ***/
        document.getElementById("button_contrast").onclick = async function(){
            console.log("Object Detection Using Yolo");
            const labels=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant", "stop sign", "parking meter", "bench", "bird","cat", "dog", "horse", "sheep", "cow","elephant", "bear", "zebra", "giraffe", "backpack","umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket", "bottle","wine glass", "cup", "fork", "knife", "spoon","bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut","cake", "chair", "couch", "potted plant", "bed","dining table", "toilet", "tv", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven","toaster", "sink", "refrigerator", "book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            const numClass=80;
            let image=document.getElementById("canva-img");
            let inputImage = cv.imread(image);//reading
            console.log("Input image width",inputImage.cols,"Input image width",inputImage.rows);
            //load the model
            model = await tf.loadGraphModel("model.json");
            //calculate model size now 
            const inputTensorShape = model.inputs[0].shape;
            const modelWidth = inputTensorShape[1];
            const modelHeight = inputTensorShape[2];
            console.log("Model width", modelWidth, "Model Height", modelHeight );
            const preprocess = (img, modelWidth, modelHeight) =>{
                let xRatio,  yRatio;
                const input = tf.tidy(()=>{
                    //Convert the Pixel Data From Image Source into TensorFLow js tensor
                    const img = tf.browser.fromPixels(image);
                    //Extracting the Width and Height of Image Tensor
                    const [h,w] = img.shape.slice(0,2);
                    //Height and Width of the Image Tensor
                    console.log("Height", h, "Width", w);
                    //Max Value Between Width and Height
                    const maxSize = Math.max(w, h);
                    console.log("max:",maxSize );
                    //Applying Padding
                    const imgPadded = img.pad([
                        [0, maxSize - h],
                        [0, maxSize - w],
                        [0,0]
                    ]);
                    xRatio = maxSize/w;
                    yRatio = maxSize/h;
                    console.log("max X:",xRatio,"max: Y",yRatio );
                    //Apply Bilinear Interpolation 
                    //(Interpolation is the process of transferring image from one resolution to another without losing image quality.)
                    return tf.image.resizeBilinear(imgPadded, [modelWidth, modelHeight]).div(255.0).expandDims(0);
                })
                return [input, xRatio, yRatio]
            };

            const [input, xRatio, yRatio] = preprocess(image, modelWidth, modelHeight);
            console.log("Input Shape", input.shape, "X-Ratio", xRatio, "Y-Ratio", yRatio);
            //Pass the Image Tensor to the Tensorflow model
            const res = model.execute(input);
            //Rearrange the Dimensions of the Tensor of our input image we passed
            //[batch_size, height, width] ==> [ batch_size, width, height];
            const transres = res.transpose([0,2,1]); // change the index
            const boxes = tf.tidy(()=>{ //clean the memory
                const w = transres.slice([0,0,2], [-1,-1,1]); // bounding box top left
                const h = transres.slice([0,0,3], [-1,-1,1]); // bounding box top left
                const x1 = tf.sub(transres.slice([0,0,0], [-1,-1,1]), tf.div(w,2));
                const y1 = tf.sub(transres.slice([0,0,1], [-1,-1,1]), tf.div(h,2));
                return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze();
                // y1, x1 = represent the bounding box GeolocationCoordinates
                // (tf.add(y1, h), tf.add(x1, w)== Y2,X2= bottom Right corner
                // squeeze for to remove single-dimensional entries from the shape of an array
            });


            //Calcualte the Confidence Score and Class Names
            const [scores, classes] = tf.tidy(() => { //transres = img tensor
                const rawScores = transres.slice([0,0,4], [-1,-1, numClass]).squeeze(0);
                return [rawScores.max(1), rawScores.argMax(1)];
            });

            //Applying Non Max supression
            // commonly used in object detection to eliminate duplicate detections
            const nms = await tf.image.nonMaxSuppressionAsync(boxes,scores,500,0.45,0.2);
            const predictionsLength = nms.size;
            console.log("Prediction Length",predictionsLength)

            if (predictionsLength > 0){
                const boxes_data = boxes.gather(nms,0).dataSync();
                //helps to gather the box info from the boxes and in sync to the java scrips file
                const score_data = scores.gather(nms, 0).dataSync();
                const classes_data = classes.gather(nms, 0).dataSync();                
                console.log("Boxes Data", boxes_data, "Score Data", score_data, "Classes Data", classes_data);
                //all the data in the form of tensors
                //but our ibounding box result in the form of 640x640 but out img size is 740x630
                const xScale = inputImage.cols/modelWidth;
                const yScale = inputImage.rows/ modelHeight;
                // now go through the each frame and draw bounding 
                console.log("Score Length: ",score_data.length);
                for (let i=0; i < score_data.length; i++){
                    const classID= classes_data[i];
                    const className = labels[classes_data[i]];
                    const ConfidenceScore = (score_data[i] * 100).toFixed(1);
                    console.log("classID: ",classID, ",className: ", className, ",ConfidenceScore: ",ConfidenceScore);
                    let [y1, x1, y2, x2] = boxes_data.slice(i*4, (i+1)*4);
                    x1 *= xRatio * xScale;
                    x2 *= xRatio * xScale;
                    y1 *= yRatio * yScale;
                    y2 *= yRatio * yScale;
                    const height = y2 - y1;
                    const width = x2 - x1;
                    console.log(x1, y1, width, height, className, ConfidenceScore);
                    let point1 = new cv.Point(x1, y1); //bounding box left corner
                    let point2 = new cv.Point(x1+ width, y1 + height); //bounding box right corner
                    cv.rectangle(inputImage, point1, point2, computeColorForLabel_s(classID), 4); 
                    const text = className + " - " + ConfidenceScore + "%"
                    // Create a hidden canvas element to measure the text size
                    const canvas = document.createElement("canvas");
                    const context = canvas.getContext("2d");
                    context.font = "22px Arial"; // Set the font size and family as needed
                    // Measure the width of the text
                    const textWidth = context.measureText(text).width;
                    cv.rectangle(inputImage, new cv.Point(x1,y1-20), new cv.Point(x1+ textWidth + context.lineWidth, y1), computeColorForLabel_s(classID),-1)
                    cv.putText(inputImage, text, new cv.Point(x1, y1 - 5),cv.FONT_HERSHEY_TRIPLEX, 0.50, new cv.Scalar(255,255,255,255), 1);
                 

                }
                cv.imshow("main-canva", inputImage);  


            }






        };


        /**** For Edge ***/
        document.getElementById("button_edge").onclick = function(){
            let imgMain = cv.imread("canva-img");//reading
            let imgCanny=imgMain.clone();
            cv.Canny(imgMain,imgCanny,50,60)
            cv.imshow("main-canva",imgCanny); //displaying at the main_canva
            imgMain.delete();
            imgCanny.delete();//we need to delete the image to free the memmory

        };
        

        /**** for Opjet Detections ***/
        document.getElementById("button_blur").onclick = function(){
            console.log("Object Dection is ON")
            const image= document.getElementById("canva-img");
            let input_image= cv.imread(image);
            // Load the model.
            cocoSsd.load().then(model => {
                // detect objects in the image.
                model.detect(image).then(predictions => {
                  console.log('Predictions: ', predictions);
                  console.log("Length of Predictions", predictions.length)
                  if (predictions.length > 0){
                    drawBoundingBox(predictions, input_image);
                    cv.imshow("main-canva", input_image)
                    input_image.delete();               
                }
                else{
                    cv.imshow("main-canva", input_image);
                    input_image.delete();
                }

                });
              });

            
        };

        


    }
}

document.getElementById('download-btn').addEventListener('click', downloadImage);








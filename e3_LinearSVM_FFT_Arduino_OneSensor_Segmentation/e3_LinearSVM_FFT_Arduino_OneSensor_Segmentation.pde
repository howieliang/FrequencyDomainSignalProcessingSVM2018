//*********************************************
// Frequency-Domain Signal Processing
// e3_LinearSVM_FFT_Arduino_OneSensor_Segmentation
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
//Before use, please make sure your Arduino has 3 sensors connected
//to the analog input, and SerialString_ThreeSensors.ino was uploaded. 
//[Mouse Right Key] Collect Data
//[0-9] Change Label to 0-9
//[ENTER] Train the SVM
//[/] Clear the Data
//[SPACE] Pause Data Stream

import processing.serial.*;
Serial port; 

int sensorNum = 1; //number of sensors in use
int dataNum = 500; //number of data to show
float sampleRate = 500;
int bufferSize = 128;

//FFT parameters
ezFFT[] fft = new ezFFT[sensorNum];
ezFFT fftMerged;
float[][] FFTHist; //history data to show
final int LOW_THLD = 2;
final int HIGH_THLD = 64; //high threshold of band-pass frequencies

//SVM parameters
double C = 64; //Cost: The regularization parameter of SVM
int d = HIGH_THLD-LOW_THLD; //to be determined
float[] modeArray = new float[dataNum]; //classification to show
int lastPredY = -1;
int maxType = 0;
double[] tempX; //Form a feature vector X;

//Global Variables for visualization
int col;
int leftedge;

//segmentation parameters
float energyMax = 0;
float energyThld = 50;
float[] energyHist = new float[dataNum]; //history data to show

//Serial Data
int[] rawData = new int[sensorNum]; //raw data from serial port
float[] postProcessedDataArray = new float[sensorNum]; //data after postProcessing
float[][] sensorHist = new float[sensorNum][dataNum]; //history data to show
float[][] diffArray = new float[sensorNum][dataNum]; //diff calculation: substract
int diffMode = 1; //0: normal diff; 1: absolute diff
int activationThld = 10; //The diff threshold of activiation
float[] activArray = new float[dataNum]; //classification to show
int windowSize = 10; //The size of data window
float[][] windowArray = new float[sensorNum][windowSize]; //data window collection
boolean b_sampling = false; //flag to keep data collection non-preemptive
int sampleCnt = 0; //counter of samples

boolean b_pause = false; //flag to pause data collection

void setup() {
  size(700, 700, P3D);
  textFont(createFont("SanSerif", 12));

  //Initiate the serial port
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }

  //ezFFT(number of samples, sampleRate)
  for (int i = 0; i < sensorNum; i++) {
    fft[i] = new ezFFT(bufferSize, sampleRate);
  }

  d = fft[0].getSpecSize()-LOW_THLD;
  FFTHist = new float[d][dataNum]; //history data to show
  tempX = new double[d]; //temp max of each freq band

  frameRate(15);
}

void draw() {
  background(255);
  stroke(0);

  energyMax = 0; //reset the measurement of energySum
  for (int i = 0; i < sensorNum; i++) {
    fft[0].updateFFT(sensorHist[i]);
    for (int j = 0; j < d; j++) {
      float x = fft[i].getSpectrum()[j+LOW_THLD];
      if(x>energyMax) energyMax = x;
      appendArrayTail(FFTHist[j], x);
      tempX[j] = x;
    }
  }
  appendArrayTail(energyHist, energyMax); //the class is null without mouse pressed.
  //drawSpectrogram();
  //Draw the modeArray
  //barGraph(float[] data, float lowerbound, float upperbound, float x, float y, float width, float height)
  barGraph(modeArray, 0, 100, 0, 700, 500, 150);
  lineGraph(energyHist, 0., height, 0., height-150, 500., 150, 0, color(0));

  ////use the data for classification
  double[] X = new double[d]; //Form a feature vector X;
  double[] dataToTrain = new double[d+1];
  double[] dataToTest = new double[d];

  if (energyMax>energyThld) {
    if (!svmTrained) { //if the SVM model is not trained
      if (mousePressed) {
        int Y = type; //Form a label Y;
        for (int i = 0; i < d; i++) {
          X[i] = tempX[i];//fft.getBand(i+LOW_THLD);
          dataToTrain[i] = X[i];
        }
        dataToTrain[d] = Y;
        trainData.add(new Data(dataToTrain)); //Add the dataToTrain to the trainingData collection.
        appendArrayTail(modeArray, Y); //append the label to  for visualization
        ++tCnt;
      }else{
        appendArrayTail(modeArray, -1); //the class is null without mouse pressed.
      }
    } else { //if the SVM model is trained
      for (int i = 0; i < d; i++) {
        X[i] = tempX[i];//= fft.getBand(i+LOW_THLD);
        dataToTest[i] = X[i];
      }
      int predictedY = (int) svmPredict(dataToTest); //SVMPredict the label of the dataToTest
      lastPredY = predictedY;
      appendArrayTail(modeArray, predictedY); //append the prediction results to modeArray for visualization
    }
  } else {
    appendArrayTail(modeArray, -1); //the class is null without mouse pressed.
  }

  ////Draw the sensor data
  ////lineGraph(float[] data, float lowerbound, float upperbound, float x, float y, float width, float height, int _index)
  //for (int i = 0; i < sensorNum; i++) {
  //  lineGraph(sensorHist[i], 0, height, 0, height-150, 500, 75, i);
  //  lineGraph(diffArray[i], 0, 100, 0, height-75, 500, 75, i); //history of diff
  //}

  for (int i = 0; i < d; i++) {
    float v = FFTHist[i][FFTHist[i].length-1];
    if (v>energyThld) lineGraph(FFTHist[i], 1000, 0, 0, (d-i)*8, 500, 10, 0, color(0, 255, 0));
    else lineGraph(FFTHist[i], 1000, 0, 0, (d-i)*8, 500, 10, 0, color(0));
  }

  if (!svmTrained && firstTrained) {
    //train a linear support vector classifier (SVC) 
    trainLinearSVC(d, C);
  }


  float fftScale = 2;
  pushMatrix();
  translate(500, 0);
  fft[0].drawSpectrogram(fftScale, 1024);
  popMatrix();

  pushStyle();
  fill(0);
  textSize(18);
  text("Threshold: "+energyThld, width-190, height-194);
  if (svmTrained) text("Last Prediction: "+lastPredY, width-190, height-170);
  else text("Current Label: "+type, width-190, height-170);
  float fps = frameRate;
  float bandwidth = sampleRate/bufferSize;
  float overlapping = (fps>0 ? 1-(sampleRate)/(bufferSize*fps):0);
  text("[N]umber of Bands: d = "+d+" [#"+LOW_THLD+"("+int(LOW_THLD*bandwidth)+"Hz)"+" - #"+HIGH_THLD+"("+int(HIGH_THLD*bandwidth)+"Hz)]", 20, height-122);
  text("[S]ample Rate: "+nf(sampleRate, 0, 0), 20, height-98);
  text("[B]uffer Size: "+bufferSize, 20, height-74);
  text("[F]PS: "+nf(fps, 0, 2), 20, height-50);
  text("Buffer Overlapping [1-S/BF]]: "+nf((float)overlapping*100, 0, 2)+" %", 20, height-26);
  popStyle();
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  int dataIndex = -1;
  if (!b_pause) {
    //assign data index based on the header
    if (inData.charAt(0) == 'A') {  
      dataIndex = 0;
    }
    //data processing
    if (dataIndex>=0) {
      rawData[dataIndex] = int(trim(inData.substring(1))); //store the value
      postProcessedDataArray[dataIndex] = rawData[dataIndex];//map(constrain(rawData[dataIndex], 0, 1023), 0, 1023, 0, height); //scale the data (for visualization)
      appendArrayTail(sensorHist[dataIndex], rawData[dataIndex]); //store the data to history (for visualization)
      float diff = sensorHist[0][sensorHist[0].length-1] - sensorHist[0][sensorHist[0].length-2]; //normal diff
      if (diffMode==1) diff = abs(diff); //absolute diff
      appendArrayTail(diffArray[0], diff);
      return;
    }
  }
}

void keyPressed() {
  if (key == ENTER) {
    if (tCnt>0 || type>0) {
      if (!firstTrained) firstTrained = true;
      resetSVM();
    } else {
      println("Error: No Data");
    }
  }
  if (key >= '0' && key <= '9') {
    type = key - '0';
    if (type>maxType) maxType = type;
  }
  if (key == 'A' || key == 'a') {
    energyThld = min(energyThld+5, 200);
  }
  if (key == 'Z' || key == 'z') {
    energyThld = max(energyThld-5, 10);
  }
  if (key == '/') {
    firstTrained = false;
    resetSVM();
    clearSVM();
    maxType = 0;
  }
  if (key == 'S' || key == 's') {
    if (model!=null) { 
      saveSVM_Model(sketchPath()+"/data/test.model", model);
      println("Model Saved");
    }
  }
  if (key == ' ') {
    if (b_pause == true) b_pause = false;
    else b_pause = true;
  }
}

//Tool functions

//Append a value to a float[] array.
float[] appendArray (float[] _array, float _val) {
  float[] array = _array;
  float[] tempArray = new float[_array.length-1];
  arrayCopy(array, tempArray, tempArray.length);
  array[0] = _val;
  arrayCopy(tempArray, 0, array, 1, tempArray.length);
  return array;
}

float[] appendArrayTail (float[] _array, float _val) {
  for (int i = 0; i < _array.length-1; i++) {  
    _array[i] = _array[i+1];
  }
  _array[_array.length-1] = _val;
  return _array;
}

//Draw a line graph to visualize the sensor stream

void lineGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h, int _index, color c) {
  pushStyle();
  float delta = _w/data.length;
  beginShape();
  noFill();
  stroke(c);
  for (float i : data) {
    float h = map(i, _l, _u, 0, _h);
    vertex(_x, _y+h);
    _x = _x + delta;
  }
  endShape();
  popStyle();
}

void lineGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h, int _index) {
  color colors[] = {
    color(255, 0, 0), color(0, 255, 0), color(0, 0, 255), color(255, 255, 0), color(0, 255, 255), 
    color(255, 0, 255), color(0)
  };
  int index = min(max(_index, 0), colors.length);
  pushStyle();
  float delta = _w/data.length;
  beginShape();
  noFill();
  stroke(colors[index]);
  for (float i : data) {
    float h = map(i, _l, _u, 0, _h);
    vertex(_x, _y+h);
    _x = _x + delta;
  }
  endShape();
  popStyle();
}

//Draw a bar graph to visualize the modeArray
void barGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), color(82, 179, 217), color(244, 208, 63), 
    color(242, 121, 53), color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), color(128, 52, 0)
  };
  pushStyle();
  noStroke();
  float delta = _w / data.length;
  for (int p = 0; p < data.length; p++) {
    float i = data[p];
    int cIndex = min((int) i, colors.length-1);
    if (i<0) fill(255, 100);
    else fill(colors[cIndex], 100);
    float h = map(_u, _l, _u, 0, _h);
    rect(_x, _y-h, delta, h);
    _x = _x + delta;
  }
  popStyle();
}
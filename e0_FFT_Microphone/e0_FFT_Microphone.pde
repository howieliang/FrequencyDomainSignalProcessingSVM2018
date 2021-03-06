//*********************************************
// Frequency-Domain Signal Processing
// e0_FFT_Microphone
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************

import ddf.minim.analysis.*;
import ddf.minim.*;

Minim minim;
AudioInput in;
FFT fft;

int dataNum = 500;
float sampleRate = 44100/5;
int bufferSize = 1024;
//FFT parameters
float[][] FFTHist;
final int LOW_THLD = 0; //low threshold of band-pass frequencies
int HIGH_THLD = 100; //high threshold of band-pass frequencies

//SVM parameters
double C = 64; //Cost: The regularization parameter of SVM 
int d = HIGH_THLD-LOW_THLD; //number of feature
float[] modeArray = new float[dataNum]; //classification to show

//Global Variables for visualization
int col;
int leftedge;

void setup()
{
  size(700, 700, P3D);
  textFont(createFont("SanSerif", 12));

  minim = new Minim(this);

  // setup audio input
  in = minim.getLineIn(Minim.MONO, bufferSize, sampleRate);

  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }

  fft = new FFT(in.bufferSize(), in.sampleRate());
  fft.window(FFT.NONE);
  
  HIGH_THLD = fft.specSize();

  d = HIGH_THLD - LOW_THLD; //for band-pass
  //d = fft.specSize()-LOW_THLD; //for high-pass
  FFTHist = new float[d][dataNum]; //history data to show
  frameRate(100);
}

void draw()
{
  
  background(255);
  stroke(0);
  // grab the input samples
  float[] samples = in.mix.toArray();
  updateFFT(samples);
  
  drawSpectrogram();
  
  pushStyle();
  fill(0);
  textSize(18);
  float fps = frameRate;
  float bandwidth = sampleRate/bufferSize;
  float overlapping = (fps>0 ? 1-(sampleRate)/(bufferSize*fps):0);
  text("[N]umber of Bands: d = "+d+" [#"+LOW_THLD+"("+int(LOW_THLD*bandwidth)+"Hz)"+" - #"+HIGH_THLD+"("+int(HIGH_THLD*bandwidth)+"Hz)]", 20, height-122);
  text("[S]ample Rate: "+nf(sampleRate,0,0), 20, height-98);
  text("[B]uffer Size: "+bufferSize, 20, height-74);
  text("[F]PS: "+nf(fps,0,2), 20, height-50);
  text("Buffer Overlapping [1-S/BF]]: "+nf((float)overlapping*100,0,2)+" %", 20, height-26);
  popStyle();
  
}

void updateFFT(float[] _samples) {
  // apply windowing
  for (int i = 0; i < _samples.length/2; ++i) {
    // Calculate & apply window symmetrically around center point
    // Hanning (raised cosine) window
    float winval = (float)(0.5+0.5*Math.cos(Math.PI*(float)i/(float)(bufferSize/2)));
    if (i > bufferSize/2)  winval = 0;
    _samples[_samples.length/2 + i] *= winval;
    _samples[_samples.length/2 - i] *= winval;
  }
  // zero out first point (not touched by odd-length window)
  _samples[0] = 0;

  // perform a forward FFT on the samples in the input buffer
  fft.forward(_samples);
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

float[] appendArrayTail (float[] _array, float _val) {
  for (int i = 0; i < _array.length-1; i++) {  
    _array[i] = _array[i+1];
  }
  _array[_array.length-1] = _val;
  return _array;
}

void drawSpectrogram() {
  int h = ceil((height-150)/(HIGH_THLD-LOW_THLD));
  // fill in the new column of spectral values  
  for (int i = 0; i < HIGH_THLD-LOW_THLD; i++) {
    //FFTHist[i][col] = Math.round(Math.max(0, 2*20*Math.log10(1000*fft.getBand(i+NUM_DC))));
    FFTHist[i][col] = fft.getBand(i+LOW_THLD);
  }
  // next time will be the next column
  col = col + 1; 
  // wrap back to the first column when we get to the end
  if (col == dataNum) { 
    col = 0;
  }

  // Draw points.  
  // leftedge is the column in the ring-filled array that is drawn at the extreme left
  // start from there, and draw to the end of the array
  for (int i = 0; i < dataNum-leftedge; i++) {
    for (int j = 0; j < HIGH_THLD-LOW_THLD; j++) {
      stroke(255-map(FFTHist[j][i+leftedge], 0, 10, 0, 255));
      //point(i, (height-150)-(j+LOW_THLD));
      float y = (j+LOW_THLD);
      line(i, (height-150)-(y*h), i, (height-150)-(y*h+(h)));
      //line(i*h, height-150-(j+LOW_THLD),i*h+(h-1), height-150-(j+LOW_THLD)); 

    }
  }
  // Draw the rest of the image as the beginning of the array (up to leftedge)
  for (int i = 0; i < leftedge; i++) {
    for (int j = 0; j < HIGH_THLD-LOW_THLD; j++) {
      stroke(255-map(FFTHist[j][i], 0, 10, 0, 255));
      float y = (j+LOW_THLD);
      line(i+dataNum-leftedge, (height-150)-(y*h), i+dataNum-leftedge, (height-150)-(y*h+(h)));
      //point(i+dataNum-leftedge, height-150-(j+LOW_THLD));
    }
  }

  // Next time around, we move the left edge over by one, to have the whole thing
  // scroll left
  leftedge = leftedge + 1; 
  // Make sure it wraps around
  if (leftedge == dataNum) { 
    leftedge = 0;
  }

  // Add frequency axis labels
  int x = dataNum + 2; // to right of spectrogram display
  stroke(0);
  line(x, 0, x, height-150); // vertical line
  fill(0);
  // Make text appear centered relative to specified x,y point 
  textAlign(LEFT, CENTER);
  for (float freq = 500.0; freq < in.sampleRate()/2; freq += 500.0) {
    int y = (height-150) - fft.freqToIndex(freq)*h; // which bin holds this frequency?
    line(x, y, x+3, y); // add tick mark
    text(Math.round(freq)+" Hz", x+5, y); // add text label
  }
  line(0, height-150, width, height-150); // vertical line
}

void stop()
{
  // always close Minim audio classes when you finish with them
  in.close();
  minim.stop();

  super.stop();
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
  }
  if (key == '/') {
    firstTrained = false;
    resetSVM();
    clearSVM();
  }
  if (key == 'S' || key == 's') {
    if (model!=null) { 
      saveSVM_Model(sketchPath()+"/data/test.model", model);
      println("Model Saved");
    }
  }
}
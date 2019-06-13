int TOTAL = 900;
int[] layer_sizes = {5, 4, 2};

Bird bird;
ArrayList<Pipe> pipes = new ArrayList<Pipe>();

PImage pipe_up_image;
PImage pipe_down_image;
PImage bird_image;

void setup() {
  fullScreen(P2D, 1);
  bird = new Bird();
  bird.brain.loadJason("data/best_bird_300.JSON");
  pipes.add(new Pipe());
  
  pipe_up_image = loadImage("data/pipe_new_2.png");
  pipe_up_image.resize(round(pipes.get(0).w), height - round(pipes.get(0).distBetweenEnds));
  pipe_down_image = loadImage("data/pipe_new_down_2.png");
  pipe_down_image.resize(round(pipes.get(0).w), height - round(pipes.get(0).distBetweenEnds));
  bird_image = loadImage("data/bird_7.png");
  bird_image.resize(round(bird.w), round(bird.h));
}

void draw() {
  background(0);
  
  bird.update();
  bird.show();
  bird.lastCall();
    
  for (Pipe p: pipes) {
    p.update();
  }
  controlSize(pipes);
  for (Pipe p: pipes) {
    p.show();
  }
  
  bird.brain.show_network(width - 800, 0, 800, 500, 50);
}
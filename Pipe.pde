class Pipe {
  float distBetweenEnds = 300;
  float speed = 5;
  float w = 150;
  float spacing = 400;
  
  float x = width;
  float yDown;
  float yUp;
  
  Pipe() {
    yUp = random(height - distBetweenEnds);
    yDown = yUp + distBetweenEnds;
    
  }
  
  boolean touching(PVector pos, float size) {
    if (pos.x + size/2 > x && pos.x - size/2 < x + w) { 
      if (pos.y + size/2 > yDown) {
        return true;
      } else if (pos.y - size/2 < yUp) {
        return true;
      } else {return false;}
    } else {
      return false;
    }
  }
  
  void update() {
    x -= speed;
  }
  
  void show() {
    imageMode(CORNER);
    image(pipe_up_image, x, yDown);
    image(pipe_down_image, x, yUp - height + round(distBetweenEnds));
  }
}

void controlSize(ArrayList<Pipe> p) {
  if (width - p.get(p.size() - 1).x - p.get(p.size() - 1).w > p.get(p.size() - 1).spacing) {
    pipes.add(new Pipe());
  }
  if (p.get(0).x + p.get(0).w < 0) {
    p.remove(0);
  }
}
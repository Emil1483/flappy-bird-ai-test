class Network {
  int[] sizes;
  int num_layers;
  double[][][] biases;
  double[][][] weights;
  double[][][] activations;
  
  Network(int[] sizes_) {
    sizes = sizes_;
    num_layers = sizes.length;
    
    biases = new double[num_layers][][];
    weights = new double[num_layers][][];
    activations = new double[num_layers][][];
    
    for (int i = 1; i < num_layers; i++) {
      biases[i] = new double[sizes[i]][1];
      biases[i] = randomize_Matrix(biases[i]);
    }
    
    for (int i = 1; i < num_layers; i++) {
      activations[i] = new double[sizes[i]][1];
    }
    
    for (int i = 1; i < num_layers; i++) {
      weights[i] = new double[sizes[i]][sizes[i-1]];
      weights[i] = randomize_Matrix(weights[i]);
    }
  }
  
  double[][] feedforward(double[][] a) {
    double[][] x = a;
    activations[0] = a;
    for (int i = 1; i < num_layers; i++) {
      x = sigmoid(matrixAdd(matrixDot(weights[i], x), biases[i]));
      activations[i] = x;
    }
    return x;
  }
  
  void mutate(float evolve_val) {
    for (int i = 1; i < num_layers; i++) {
      biases[i] = evolve_matrix(biases[i], evolve_val);
    }
    
    for (int i = 1; i < num_layers; i++) {
      weights[i] = evolve_matrix(weights[i], evolve_val);
    }
  }
  
  void copy_brain(Network nn) {
    sizes = nn.sizes;
    num_layers = nn.sizes.length;
    
    biases = new double[num_layers][][];
    weights = new double[num_layers][][];
    
    for (int i = 1; i < num_layers; i++) {
      biases[i] = new double[sizes[i]][1];
      biases[i] = nn.biases[i];
    }
    
    for (int i = 1; i < num_layers; i++) {
      weights[i] = new double[sizes[i]][sizes[i-1]];
      weights[i] = nn.weights[i];
    }
  }
  
  void load_type_weights(int i, int j, int k) {
    float aw = (float)activations[i-1][k][0] * abs((float)weights[i][j][k]);
    float alpha = abs(map(aw, 0, 1, 0, 255));
    float weights_map = map(round((float)weights[i][j][k]), -1, 1, 0, 255);
    float strokeWeight = map(abs(aw), 0, 1, 1, 8);
    strokeWeight(strokeWeight);
    if (activations[i-1][k][0] > 0) {
      stroke(255 - weights_map, 0, weights_map, alpha);
    } else {
      stroke(weights_map, 0, 255 - weights_map, alpha);
    }
  }
  
  void load_type_nodes(int i, int j) {
    if (i >= 1) {
      float strokeWeight = map(abs((float)biases[i][j][0]), 0, 1, 0, 5);
      float biases_map = map(round((float)biases[i][j][0]), -1, 1, 0, 255);
      strokeWeight(strokeWeight);
      stroke(255 - biases_map, 0, biases_map);
    } else { noStroke(); }
    if (i == num_layers-1) {
      int diff;
      if (j == 0) { diff = 1; } else { diff = -1; }
      if (activations[i][j][0] > activations[i][j+diff][0]) 
        { fill(0, 0, 255); } else { fill(0); }
    } else {
      fill(map((float)activations[i][j][0], -1, 0, 255, 0), 0, map((float)activations[i][j][0], 0, 1, 0, 255));
    }
  }
  
  void show_network(float x, float y, float w, float h, float size) {
    fill(0);
    stroke(0, 0, 170);
    rect(x, y, w, h);
    for (int i = 1; i < num_layers; i++) {
      for (int j = 0; j < sizes[i]; j++) {
        for (int k = 0; k < sizes[i - 1]; k++) {
          load_type_weights(i, j, k);
          float x1 = size/2 +  i      * (w-size) / (num_layers-1);
          float y1 = size/2 +  j      * (h-size) /   (sizes[i]-1);
          float x2 = size/2 + (i - 1) * (w-size) / (num_layers-1);
          float y2 = size/2 +  k      * (h-size) / (sizes[i-1]-1);
          line(x + x1, y + y1, x + x2, y + y2);
        }
      }
    }
    
    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < sizes[i]; j++) {
        load_type_nodes(i, j);
        ellipse(x + size/2 + i * (w-size) / (num_layers-1), y + size/2 + j * (h-size) / (sizes[i]-1), size, size);
        textSize(20);
        fill(0, 255, 0);
      }
    }
  }
  
  void saveJason(String directory) {
    JSONObject json = new JSONObject();
    for (int i = 1; i < num_layers; i++) {
      for (int j = 0; j < sizes[i]; j++) {
        json.setDouble("biases[" + str(i) + "][" + str(j) + "][0]", biases[i][j][0]);
        for (int k = 0; k < sizes[i - 1]; k++) {
          json.setDouble("weights[" + str(i) + "][" + str(j) + "][" + str(k) + "]", weights[i][j][k]);
        }
      }
    }
    saveJSONObject(json, directory);
  }
  
  void loadJason(String directory) {
    JSONObject json = loadJSONObject(directory);
    for (int i = 1; i < num_layers; i++) {
      for (int j = 0; j < sizes[i]; j++) {
        biases[i][j][0] = json.getDouble("biases[" + str(i) + "][" + str(j) + "][0]");
        for (int k = 0; k < sizes[i - 1]; k++) {
          weights[i][j][k] = json.getDouble("weights[" + str(i) + "][" + str(j) + "][" + str(k) + "]");
        }
      }
    }
  }
}
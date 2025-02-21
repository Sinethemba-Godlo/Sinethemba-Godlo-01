use rand::Rng;

struct LinearRegressionModel {
    weight: f64, // Equivalent to slope
    bias: f64,   // Equivalent to y-intercept
    learning_rate: f64,
}

impl LinearRegressionModel {
    fn new(learning_rate: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            weight: rng.random::<f64>(),
            bias: rng.random::<f64>(),
            learning_rate,
        }
    }

    fn predict(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    fn train(&mut self, x_values: &[f64], y_values: &[f64], epochs: usize) {
        let n = x_values.len() as f64;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut weight_gradient = 0.0;
            let mut bias_gradient = 0.0;

            for (&x, &actual_y) in x_values.iter().zip(y_values.iter()) {
                let predicted_y = self.predict(x);
                let error = predicted_y - actual_y;

                // Compute gradients
                weight_gradient += error * x;
                bias_gradient += error;
                total_loss += error * error;
            }

            // Update parameters using gradient descent
            self.weight -= (self.learning_rate * weight_gradient) / n;
            self.bias -= (self.learning_rate * bias_gradient) / n;

            // Print loss every 100 epochs
            if epoch % 100 == 0 {
                println!("Epoch {}, Loss: {:.4}", epoch, total_loss / n);
            }
        }
    }

    fn print_model(&self) {
        println!("Final model: y = {:.4}x + {:.4}", self.weight, self.bias);
    }
}

struct DataGenerator;

impl DataGenerator {
    fn generate_data(num_points: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = rand::rng();
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for _ in 0..num_points {
            let x = rng.random_range(0.0..=10.0);
            let y = 2.0 * x + 1.0 + rng.random::<f64>() * 0.2;
            x_values.push(x);
            y_values.push(y);
        }

        (x_values, y_values)
    }
}

fn main() {
    // Generate training data
    let (x_values, y_values) = DataGenerator::generate_data(100);

    // Create and train the model
    let mut model = LinearRegressionModel::new(0.01);
    model.train(&x_values, &y_values, 1000);
    model.print_model();

    // Test model with sample points
    let test_points = [1.0, 2.0, 3.0, 4.0, 5.0];
    println!("\nModel predictions:");
    for &x in &test_points {
        let predicted = model.predict(x);
        let expected = 2.0 * x + 1.0;
        println!("x: {:.1}, Predicted: {:.4}, Expected: {:.4}", x, predicted, expected);
    }
}

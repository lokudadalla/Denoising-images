# Denoising-images
denoising images using dental image dataset

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
   
</head>
<body>
    <div class="container">
        <h1>Denoising Grayscale Images Using U-Net</h1>
        <p>This project focuses on using a U-Net convolutional neural network to denoise grayscale dental images. The workflow involves several key steps: loading the dataset, adding synthetic noise, training the U-Net model, and evaluating its performance.</p>

  <h2>Image Loading and Preprocessing</h2>
        <p><strong>Functions for Reading Images:</strong></p>
        <p>1. <strong>read_dental function:</strong></p>
        <ul>
            <li>Uses OpenCV to read images from a specified folder.</li>
            <li>Converts images to grayscale.</li>
            <li>Stores images in a numpy array.</li>
        </ul>
        <p>2. <strong>read_all_datasets function:</strong> Calls <code>read_dental</code> to get the dataset.</p>

   <h2>U-Net Model Definition</h2>
        <p>The <code>unet_model</code> function defines a U-Net architecture. U-Net is a type of convolutional neural network particularly well-suited for image segmentation and denoising tasks due to its symmetric encoder-decoder structure with skip connections. The encoder downsamples the input image while capturing context, and the decoder upsamples it to restore the original size, using skip connections to retain spatial information. This architecture helps in effectively reconstructing denoised images.</p>
        <p>Consists of convolutional and max-pooling layers to downsample the image and capture context. The bottleneck is the deepest part of the network that captures high-level features. Upsampling and convolutional layers are used to reconstruct the image, using skip connections from the encoder to preserve spatial information.</p>

  <img src="https://private-user-images.githubusercontent.com/133969661/333787105-1f7a3ff5-9e37-4e01-b8d5-4d4a2119bb55.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjM5MTAxODEsIm5iZiI6MTcyMzkwOTg4MSwicGF0aCI6Ii8xMzM5Njk2NjEvMzMzNzg3MTA1LTFmN2EzZmY1LTllMzctNGUwMS1iOGQ1LTRkNGEyMTE5YmI1NS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgxN1QxNTUxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iMjMxM2ZhMGUyODk4Y2UwMTRjNWJkOGI5MDA1MmQ3NTc5M2I2MGYyZDIzODBjOWE5ZDgwZmRlNGQwYjM3ODIyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.sYLpCl3Da4f2TJ3YyfUGfeWZQDXwGlvv8qv1to04Ap4" alt="Noisy Image">
   <img src="https://private-user-images.githubusercontent.com/133969661/333787108-5c547235-fee5-45d7-90ef-101d04c0b29e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjM5MTAxODEsIm5iZiI6MTcyMzkwOTg4MSwicGF0aCI6Ii8xMzM5Njk2NjEvMzMzNzg3MTA4LTVjNTQ3MjM1LWZlZTUtNDVkNy05MGVmLTEwMWQwNGMwYjI5ZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgxN1QxNTUxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NTIyOTgwZDkxZTA0MDNiYWYzYjZhMWEzNzBmZjQyOGI0ZDkwZmNkMmIyN2I0MzQyOTQ2MTc4YmM1ZDJhYzEyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.1c4Tfbc4OVEz3glkca4cgBhVDnOt6z3vtqb-wq4MX_4" alt="Noisy Image">

   <h2>Training and Evaluation</h2>
        <p>A <code>CNN_denoiser</code> class is defined to manage the U-Net model's training and evaluation. The constructor initializes the model with specified parameters such as batch size, number of epochs, and validation split. The training method trains the model using noisy and clean images, while the evaluation method assesses its performance on the test set, printing the loss and accuracy.</p>

  <img src="https://private-user-images.githubusercontent.com/133969661/333787110-ef67e6ac-48e1-4bd7-9e47-7d9314abccd1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjM5MTAxODEsIm5iZiI6MTcyMzkwOTg4MSwicGF0aCI6Ii8xMzM5Njk2NjEvMzMzNzg3MTEwLWVmNjdlNmFjLTQ4ZTEtNGJkNy05ZTQ3LTdkOTMxNGFiY2NkMS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgxN1QxNTUxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kZGZiMTk5MzNhYjM1ZTc5YjViZDYxOTQzNGUyMTRkYzkxZDI5MDQ2ZTQyNjBlMzQzYjRjOTAxM2Q5ODJkZWQwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.cIp3H0XLO86mch7M0Xrn9cpxIv0kXRlWBR6IckewfXQ" alt="Noisy Image">

   <h2>Adding Noise and Training the Model</h2>
        <p>The main block of the code reads the dataset, resizes the images, and normalizes them. It then splits the dataset into training and test sets, adds Gaussian noise to simulate noisy conditions, and trains the U-Net model using these noisy images. The denoised images generated by the model are plotted for comparison with the noisy inputs.</p>

   <p>I trained the U-Net model for only 10 epochs to generate the current denoised images from the noisy images. Below, you can see the comparison between the noisy input images and the denoised output images produced by the model.</p>

   <img src="https://private-user-images.githubusercontent.com/133969661/333787112-5d1fd9c7-26a0-4b32-97ab-95f6892c95c3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjM5MTAxODEsIm5iZiI6MTcyMzkwOTg4MSwicGF0aCI6Ii8xMzM5Njk2NjEvMzMzNzg3MTEyLTVkMWZkOWM3LTI2YTAtNGIzMi05N2FiLTk1ZjY4OTJjOTVjMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwODE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDgxN1QxNTUxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yYWEyMGFkNjc5NTA1N2I4MTJiYjE5OTU5NGU1NzY5NmQ0MDc2OTliZGIyZWJmNDg2ZWI4MzljOTM5YzNjOWM0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.xJlwwm893IkpN7yJ8HbfPS2Vn3eStgw3HQKwSaJKU0w" alt="Noisy Image">

  <p>While the current results are promising, it's important to note that training the model for more epochs, such as 100 epochs or more, is expected to significantly improve the results. With extended training, the model has more opportunities to learn and fine-tune its weights, leading to higher accuracy and lower loss. Consequently, the denoised images will exhibit better quality, with reduced noise and enhanced details.</p>

<h2>Resources</h2>
<a href="https://arxiv.org/pdf/1608.04667v2">https://arxiv.org/pdf/1608.04667v2</a>


  
</div>
</body>
</html>

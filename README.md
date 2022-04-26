# Inpainting
This in an implementation of the article [Region Filling and Object Removal by Exemplar-Based Image Inpainting](https://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf) as part of the *Image inpainting project* for IMA201 (Télécom Paris).

<br/>

***

### Running the program
Place yourself in the "inpainting" directory, then execute :

```python Code/main.py -i [path/to/image] -m [path/to/mask] -o [path/to/output/image] ```

<br/>

By default, ```python Code/main.py``` will take *Image.png* and *Mask.png* in Data folder, and then save output image as *Output.png* in the same folder.

***
### Examples

<p float="center">
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Baseball_process.gif" width=248/>
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Square_process.gif" width=248/>
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Island_process.gif" width=248/>
  
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Bike_process.gif" width=375 height = 240/>
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Guys_process.gif" width=375 height=240/>
  
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Flower_process.gif" width=375 height=240/>
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Wall_process.gif" width=375 height=240/>
  
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Eiffel_process.gif" width=375 height=240/>
  <img src="https://github.com/GabFrancon/inpainting/blob/main/Data/Process/Old_process.gif" width=375 height=240/>
</p>

***

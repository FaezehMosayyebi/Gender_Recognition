# Official code repository for "Gender recognition in masked facial images using EfficinetNet and transfer learning approach"

<p>The rapid advancement of Artificial Intelligence (AI) technology has enabled diverse systems to leverage 
individual characteristics to enhance their functionality. Among these attributes, gender information is 
pivotal in human-machine interaction across various domains, such as vending machines and targeted 
advertising campaigns. While numerous methodologies have been developed for gender detection from facial 
images, they often encounter limitations, particularly in scenarios involving masked individuals during 
unprecedented events like the COVID-19 pandemic. This study tries to present an optimal Convolutional 
Neural Network architecture for gender recognition based on EfficientNet, which is recognized as the most 
effective backbone network for gender recognition during some experiments, and train it with a masked-worn 
faces database to yield an efficient network for detecting the gender of masked-worn people. However, 
creating such datasets proves to be both time-consuming and resource-intensive. Using the Poisson Image 
Editing technique, this article introduces an innovative and cost-efficient approach to generating masked 
facial images from pre-existing face databases to surmount this obstacle. Remarkably, the accuracy achieved 
by the proposed methodologies on three renowned databases—LFW, CelebA, and Adience— for original images is 
an impressive 98.5%, 98.27%, and 95.80%, respectively. This outstanding performance demonstrates a significant 
advancement over prior endeavors, underscoring the efficiency and robustness of our approach.</p> 

[Link to paper](https://link.springer.com/article/10.1007/s41870-023-01565-4)

# How to use codes:

1. Clone codes.
```
git clone https://github.com/FaezehMosayyebi/Gender_Recognition.git
```
2. Install requirements.

```
python3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
3. Run interaction.ipynb


# Citation
When you publish your research using these codes, please cite [1] as
<br />
<br />
@article{
<br />
  title = {Gender recognition in masked facial images using EfficientNet and transfer learning approach,
  <br />
  author = {Mosayyebi, Faezeh, Seyedarabi, Hadi, and Afrouzian, Reza},
  <br />
  journal = {International Journal of Information Technology},
  <br />
  volume = {},
  <br />
  pages = {},
  <br />
  year = {2023},
  <br />
  doi = {10.1007/s41870-023-01565-4},
  <br />
  publisher = {Springer},}
<br />

# References
[1] F. Mosayyebi, H. Seyedarabi, and R. Afrouzian, "Gender recognition in masked facial images using EfficinetNet and transfer learning approach," International Journal of Information Technology, 2023/10/20 2023, doi: 10.1007/s41870-023-01565-4.
<br />

Contact: faezeh.mosayyebi@gmail.com
<br />

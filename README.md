# Image rendering style transfer 


### **Goal**: 

Give a z-stack (data cube) and a target style, adjust all the parameters such that the resulting image has the same style. 


### **Definitions**:
**Data cube (d)**: 3D stack of images with intensity values. 

**Rendered image (r)**: 2D visualization of a data cube 

**Render style (s)**: quantitative representation of a rendered image

**Rendering parameters(Ѱ)**: all adjustable settings, ie brightness, transfer function, color, path trace etc. 


### **Functions**: 


**Render**: take a data cube and some parameters and create a 2D image. (QUESTION, shouldn’t camera settings also be in this function?)

 *f<sub>render</sub>(d, Ѱ) = r*


**Ref**: take a 2D image and create a numeral representation. 


*f<sub>style</sub>(r) = s*

**Predict**: given a data cube and a target render style, return the parameter settings. 

*f<sub>predict</sub>(d, s) = Ѱ*




### **Method**:



1. Train a neural net to be able to distinguish between render styles. IE given two different camera angle renderings of the same data cube with same rendering parameters return “same”/ 0. 
    1. For each data cube: 
        1. Choose a random set* of parameters. Take many images with the same parameters, but different camera angles. (Probably need to limit the angle to top of the data cube because of non isotropic nature of the data)
        2. Calc **s** for each** r**. 
    2. Loss function:  (s<sub>1a</sub> - s<sub>1b</sub>)<sup>2</sup> - (s<sub>1a</sub>- s<sub>2a</sub>)<sup>2 </sup>. The difference between the **s** resulting from the same data cube (1)  should be small, therefore the first term should be small. The difference between the **s **resulting from two different data cubes (1 versus 2) should be large. Therefore this function should be a large negative number. 


### **Result:**

Given d<sub>1</sub>, d<sub>2</sub> 

Create a target render image with some settings

f_render(d_1,psi_1)=r_target

f_style(r_target) = s_target


Find settings (**Ѱ<sub>2</sub>**) for d<sub>2</sub> where the resulting image will look the same as target from d<sub>1</sub>
f_predict(d_2, s_target) = Ѱ_2 

    
	f_style(f_render(d_2,Ѱ_2)) = s_target

### **Questions/Things to do:**

- How many data cubes do we need?

- How many images of each data cube do we need?

- Define the camera restrictions.

- How will we determine the parameters for the training set? How random is random?

- How many control points on the transfer function
- Color: hue saturation and brightness?

- Automate the production of a training set?

- How do we create the numeral representation of the 2D images, ie f<sub>ref</sub>?

- Generate on the fly vs. beforehand: iteration loop is 0.5 - 2 secs. 

	

**main loop:**


```
# n is batch size
# im_cube is a list of length n, each element being a c by z by y by x Tensor (data cubes)
# im_2d is a list of length n, each element being a list of m Tensors of dimensions c by y by x (rendered images)
# psi is a list of length n, each element being a set of render parameters (other than camera angle)

for i_batch, sample_batched in enumerate(dataloader):

	optim.zero_grad()
	# we associate a group of 2d images with each cube and psi
	im_cube, im_2d, im_2d_cube_id, psi = sample_batched

	style = f_style(torch.concat(im_2d))
	psi_hat = f_psi(im_cube, style)

	#alternatively
	# psi_hat = f_psi(im_cube, f_style(im2d))

	loss_psi = loss_fn(psi_hat, psi)

	#this could be sped up by randomly shuffling things
	loss_style = torch.zeros(1)
	for i, s in enumerate(style):
		list_of_same_ids = torch.find(im_2d_cube_id[i] == im_2d_cube_id)
		list_of_different_ids = torch.find(im_2d_cube_id[i] != im_2d_cube_id)
		id_same = randomly_choose(list_of_same_ids)
		Id_different = randomly_choose(list_of_different_ids)

		loss_style = loss_style + (s - style[id_same])**2 - (s - style[id_different])**2

	loss_style = loss_style/len(style)
	total_loss = loss_psi + lambda*loss_style

	total_loss.backward()

	logger.log(loss_psi, loss_style)

	optim.step()
```


	

Alternative Main loop that might not work


```
# n is batch size
#im_cube is a n by c by z by y by x Tensor
#im_2d is an n by c by y by x Tensor
#psi is an n by #parameters Tensor

for i_batch, sample_batched in enumerate(dataloader):

	optim.zero()

	im_cube, im_2d, psi = sample_batched

	style = f_style(im_2d)
	psi_hat = f_psi(im_cube, style)

	#alternatively
	# psi_hat = f_psi(im_cube, f_style(im2d))

	error = loss_fn(psi_hat, psi)

	error.backward()

	optim.step()

	logger.log(error)


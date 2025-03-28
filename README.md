# Ray Tracing Emulator @ TAU

This project generates stunning 3D scenes using a ray tracing algorithm. Given a simple scene description file, the renderer produces realistic images with lighting, shadows, and reflections.


Here’s an example of a rendered scene:

![image](https://github.com/user-attachments/assets/9befa618-e0cd-471f-90d7-9ceb03ba93f3)

Which is made from a textual scene description file:

```
# Camera: 	px   	py   	pz 	lx  	ly  	lz 	ux  	uy  	uz 	sc_dist	sc_width
cam 	  	0    	0	-2.8 	0   	0   	0  	0   	1   	0  	1	1
# Settings: 	bgr  	bgg  	bgb	sh_rays	rec_max SS
set 		1  	1  	1   	1 	10	1

# Material:	dr    	dg    	db	sr   	sg   	sb 	rr   	rg  	rb	phong 	trans
mtl		0.95	0.5	0.4	0.3	0.3	0.3	0	0	0	4	0
mtl		0.4	0.95	0.4	0.3	0.3	0.3	0	0	0	4	0
mtl		0.4	0.5	0.95	0.3	0.3	0.3	0	0	0	4	0
mtl		0.6	0.4	0.9	0.3	0.3	0.3	0	0	0	4	0
mtl		0.7	0.7	0.7	0.3	0.3	0.3	0	0	0	4	0
mtl		0.24	0.22	0.22	0.7	0.7	0.8	0	0	0	100	0
mtl		0.5	0.5	0.8	0.5	0.5	0.8	0	0.3	0.4	100	0


# Plane:	nx	ny	nz	offset	mat_idx
pln		0	1	0	-1	1
pln		0	-1	0	-1	2
pln		1	0	0	-1	3
pln		-1	0	0	-1	4
pln		0	0	-1	-1	5
pln		0	0	1	-3      5

# Sphere:	cx   	cy   	cz  	radius 	mat_idx
sph		0.5	-0.7	0.6	0.3	6
sph		-0.3	-0.5	-0.5	0.2	7

# Lights:	px	py	pz	r	g	b	spec	shadow	width
lgt		0	0	0	0.5	0.5	0.5	1	0.5	0
lgt		0	0	-1	0.5	0.5	0.5	1	0	0
#lgt		-0.9	-0.9	-0.9	0.8	0.8	0.8	1	0	0
```

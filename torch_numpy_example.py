import numpy as np
from scipy.stats import linregress

x = np.linspace(-2, 2, 10)
y = -0.5 + 0.7*x

a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = .01
equation_strings = []
loss_history = []
add_another_param = False

for t in range(30):
	y_pred = a
	if add_another_param:
		y_pred = a + b*x
	# y_pred = a + b*x + c*x**2
	# y_pred = a + b*x + c*x**2 + d*x**3
	
	loss = np.square(y_pred - y).sum()

	
	grad_y_pred = 2.0 * (y_pred - y)
	grad_a = grad_y_pred.sum()
	grad_b = (grad_y_pred * x).sum()
	grad_c = (grad_y_pred * x ** 2).sum()
	grad_d = (grad_y_pred * x ** 3).sum()
	
	a -= learning_rate * grad_a
	b -= learning_rate * grad_b
	c -= learning_rate * grad_c
	d -= learning_rate * grad_d
	
	print(f'Loss: {loss}')
	if not add_another_param:
		equation_strings.append(f'y = {a}\n')
	else:
		equation_strings.append(f'y = {a} + {b}x\n')
	# equation_strings.append(f'y = {a} + {b}x + {c}x^2\n')
	# equation_strings.append(f'y = {a} + {b}x + {c}x^2 + {d}x^3\n')
	
	# Calculating if we are hitting a local minima by seeing if loss is staying constant
	loss_history.append(loss)
	if len(loss_history) > 1:
		loss_x_values = [x for x in range(len(loss_history) * -1 + 1, 1)]
		slope = linregress(loss_x_values, loss_history).slope
		print("Loss slope: " + str(slope))
		if slope > -1:
			add_another_param = True
	
equation_file = open("equation-strings.txt", "w")
equation_file.writelines(equation_strings)
equation_file.close()
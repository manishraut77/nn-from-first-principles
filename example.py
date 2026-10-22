# example.py
from neurons import MLP



# Initialize a neural network: 3 inputs → 5+5+3 hidden neurons → 1 output
model = MLP(3, [5,5,3,1])

xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
targets=[1.0,-1.0,-1.0,1.0]

for k in range(500):
    # forward pass
    ypred=[model(x)[0] for x in xs]
    loss=sum((yp-ya)**2 for yp,ya in zip(ypred,targets))


    #backward pass

    for p in model.parametres():
        p.grad=0

    loss.backward()

    # upadate the weights

    for p in model.parametres():
        p.value+=-0.01*p.grad

    print("Step : " ,k, "Loss:",loss.value)

for pred in ypred:
    print(pred.value)

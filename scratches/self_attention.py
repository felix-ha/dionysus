import torch
import torch.nn as nn
from torch.nn import functional as F


B, T, C = 1, 3, 2

X = torch.ones([T, C])

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)

print("Straight forward mean")
print(X)
print(wei)
print(wei @ X)


tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ X

print("\nSofmax Mean")
print(X)
print(wei)
print(wei @ X)



#-------------------------------------------------
print("\n \nSelf attention")
B, T, C = 1, 5, 4
head_size = 3
X = torch.rand([B, T, C])

key = nn.Linear(C, head_size, bias=False)   # C x head_size         
query = nn.Linear(C, head_size, bias=False) # C x head_size 
value = nn.Linear(C, head_size, bias=False) # C x head_size 

k = key(X)    # B x T x head_size
q = query(X)  # B x T x head_size
v = value(X)  # B x T x head_size

print("\n X")
print(X)

print("query")
print(query.weight)

print("q = query(X)")
print(q)

# wei needs to be B x T x T

wei = q @ k.transpose(-2, -1) * head_size**-0.5# B x T x head_size @ B x head_size x T --> B x T x T

tril = torch.tril(torch.ones(T, T))

print(wei)

wei = wei.masked_fill(tril == 0, float('-inf'))

print(wei)

wei = F.softmax(wei, dim=-1)

print(wei)

out = wei @ v

print("\nv")
print(v)

print("\nout = wei @ v")
print(out)


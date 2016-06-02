function bw=get_map(n,c)
X=linspace(0,1,2*n-1);
bw=[];
for x=X
    r=x;
    y=(1-r)*[1 1 1]+r*c;
    bw=[bw;y];
end

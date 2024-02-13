function expr = SIR(t,in2,in3)
%SIR
%    EXPR = SIR(T,IN2,IN3)

%    This function was generated by the Symbolic Math Toolbox version 9.3.
%    20-Nov-2023 16:19:28

I = in2(3,:);
R = in2(1,:);
S = in2(2,:);
param1 = in3(:,1);
param2 = in3(:,2);
t2 = I.*param2;
t3 = I+R+S;
t4 = 1.0./t3;
expr = [t2;-I.*S.*param1.*t4;-I.*t4.*(t2+R.*param2-S.*param1+S.*param2)];
end

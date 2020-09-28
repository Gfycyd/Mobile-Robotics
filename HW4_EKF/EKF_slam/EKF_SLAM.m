%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  16833 Robot Localization and Mapping  %
%  Assignment #2                         %
%  EKF-SLAM                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

%==== TEST: Setup uncertainity parameters (try different values!) ===
sig_x = 0.25;
sig_y = 0.1;
sig_alpha = 0.1;
sig_beta = 0.01;
sig_r = 0.08;

%==== Generate sigma^2 from sigma ===
sig_x2 = sig_x^2;
sig_y2 = sig_y^2;
sig_alpha2 = sig_alpha^2;
sig_beta2 = sig_beta^2;
sig_r2 = sig_r^2;

%==== Open data file ====
fid = fopen('../data/data.txt');

%==== Read first measurement data ====
tline = fgets(fid);
arr = str2num(tline);
measure = arr';
t = 1;
 
%==== Setup control and measurement covariances ===
control_cov = diag([sig_x2, sig_y2, sig_alpha2]);
measure_cov = diag([sig_beta2, sig_r2]);

%==== Setup initial pose vector and pose uncertainty ====
pose = [0 ; 0 ; 0];
pose_cov = diag([0.02^2, 0.02^2, 0.1^2]);

%==== Setup initial landmark vector landmark[] and covariance matrix landmark_cov[] ====
%==== (Hint: use initial pose with uncertainty and first measurement) ====

number_of_landmarks = 6;
landmark = zeros(number_of_landmarks*2,1);
landmark_cov = zeros(number_of_landmarks*2,number_of_landmarks*2);

% for vizualization in report and debugging:
% landmark_cov = diag([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]);

for i = 1:number_of_landmarks
    landmark(2*i-1:2*i,1)= [measure(2*i)*cos(measure(2*i-1));measure(2*i)*sin(measure(2*i-1))];
    R = [-measure(2*i)*sin(measure(2*i-1)) cos(measure(2*i-1)); measure(2*i)*cos(measure(2*i-1)) sin(measure(2*i-1))];
    landmark_cov(2*i-1:2*i, 2*i-1:2*i) = R*measure_cov*R;
end
%==== Setup state vector x with pose and landmark vector ====
x = [pose ; landmark];

%==== Setup covariance matrix P with pose and landmark covariances ====
P = [pose_cov zeros(3, 2*number_of_landmarks) ; zeros(2*number_of_landmarks, 3) landmark_cov];

%==== Plot initial state and conariance ====
last_x = x;
drawTrajAndMap(x, last_x, P, 0);

%==== Read control data ====
tline = fgets(fid);
while ischar(tline)
    arr = str2num(tline);
    d = arr(1);
    alpha = arr(2);
    at = x(3);
    
    %==== Predict Step ====
    %==== (Notice: predict state x_pre[] and covariance P_pre[] using input control data and control_cov[]) ====
    F = zeros(3+2*number_of_landmarks,3);
    F(1:3,1:3)  = eye(3);

    x_pre = x + F*[d*cos(at);d*sin(at); alpha];
    F_p = [[1 0 -d*sin(alpha); 0 1 d*cos(alpha); 0 0   1] zeros(3, 2*number_of_landmarks); zeros(2*number_of_landmarks, 3) eye(2*number_of_landmarks)];

    P_pre =  F_p*P*F_p' + F*control_cov*F';
    %==== Draw predicted state x_pre[] and covariance P_pre[] ====
    drawTrajPre(x_pre, P_pre);
    
    %==== Read measurement data ====
    tline = fgets(fid);
    arr = str2num(tline);
    measure = arr';


    %==== Update Step ====
    for i=1:2:2*number_of_landmarks-1
        
        delta_x = x_pre(3+i)-x_pre(1);
        delta_y = x_pre(3+i+1)-x_pre(2);
        delta = [delta_x;delta_y];
        q=(delta'*delta)^0.5;
        beta_pred = wrapToPi(atan2(delta_y, delta_x) - x_pre(3));
        z_pred = [q;beta_pred];
        
        F = zeros(5,3+2*number_of_landmarks);
        F(1:3,1:3)=eye(3);
        F(4:5, 3+i:3+i+1)=eye(2);
        
        
        H=[-q*delta_x,-q*delta_y,0,q*delta_x,q*delta_y; delta_y,-delta_x,-(q^2),-delta_y,delta_x];
        H=H/(q^2);
        H=H*F;
        
        temp = inv(H*P_pre*H' + measure_cov);
        K=P_pre*H'*temp;
        
        z_actual = [measure(i+1);measure(i)];
        
        x_pre = x_pre + K*(z_actual-z_pred);
        P_pre= (eye(3+2*number_of_landmarks)-K*H)*P_pre;
        
          
    
    end

    x = x_pre;
    P = P_pre;

    %==== Plot ====
    drawTrajAndMap(x, last_x, P, t);
    last_x = x;
    
    %==== Iteration & read next control data ===
    t = t + 1;
    tline = fgets(fid);
end
%==== EVAL: Plot ground truth landmarks ====


% plot true landmarks
ground_truth = [3 6; 3 12; 7 8; 7 14; 11 6; 11 12];
scatter(ground_truth(:,1), ground_truth(:,2), '*k');



% calculate the euclidean distances
landmarks_predicted = reshape(x(4:end), 2, k)';
euclidean_distance = sqrt(sum((ground_truth - landmarks_predicted).^ 2, 2));

disp(euclidean_distance)


% calculate the Mahalanobis distances

mahalanobis_distance = [0 0 0 0 0 0];

for i = 1:number_of_landmarks
    b_1_index = ground_truth(i,1)-x(3+i*2-1);
    b_2_index = ground_truth(i,2)-x(3+i*2);
    b = [b_1_index; b_2_index];
    mahalanobis_distance(i) = sqrt(b'*P(3+2*i-1:3+2*i,3+2*i-1:3+2*i)*b);

end
disp(mahalanobis_distance)

%==== Close data file ====
fclose(fid);

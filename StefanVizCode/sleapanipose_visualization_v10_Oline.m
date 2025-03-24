% This script imports camera calibration parameters and matched pairs for
% objects in 3D space, and outputs a 3D projection.

% v02 integrates static skeleton of the wedge cage
% v04 corrects an error in the matrix rotation
% v05 correctdefinitions of the 4 corners of the playpen
% v06 Change how the center of the playpen is defined and how translation uses that info
% v09 Updates how videos are written to disk
% v10 Syntax updates 

clear
close all
clc


%% Session inputs
%**************************************************************************
% Camera parameters
FR = 60;
FR_subsample = 4; % 2 = every other frame
slomoFactor=1;

% Video duration
v_start = 0*FR+1; % seconds converted to frames
v_duration = 10 - 1/FR; % seconds
v_stop = v_start+FR*v_duration-1; % seconds converted to frames
numFrames_output = (v_stop - v_start)+1; % Calculate output frames

%**************************************************************************
% Skeletons
numAnimals = 3;
numStatics = 1;

% Define mouse data
points_mouse = h5read('/Volumes/falkner/Meenakshi/Videos/2025-02-11/CD1F15/20250211_162736-CD1F15_CSI/points3d_sSmooth_3_RPE_15_50400_57600.h5','/tracks');
Mice = permute(points_mouse,[4, 2, 1, 3]);

numFrames = size(Mice,1);
numNodes = size(Mice,2);

% Define skeleton
%skeleton_file  = 'skeleton_3D_SNO_17node_noTail_v02.mat';
%skeleton_file  = 'skeleton_SNO_34node_v09.mat';
skeleton_file  = 'skeleton_SNO_15node.mat';

% Define static data
points_cage = h5read('/Volumes/falkner/Meenakshi/Videos/2025-02-11/CD1F15/statics/points3d_static.h5','/tracks');

points_cage = permute(points_cage,[4, 2, 1, 3]);
points_cage(1,:,:) = median(points_cage,1);

numNodesCage = size(points_cage,2);
static_reference = 1;
metric_conversion_flag = 1;

%**************************************************************************
% Plotting parameters
% Correct for Windows/OSX monitor scaling (Mac Retina displays need high values
% like 200%)
monitor_scaling = 100; %Typical values are 100%, 125%, 150%

% Resolution adjustment
video_resolution_adjustment = 1; 

% Manual video scale factor
video_manual_scale = 1;

% Scale the total size of the video
video_size_coeff = video_resolution_adjustment * ((1/monitor_scaling)*100) * video_manual_scale;

% Plot text for cardinal directions?
plot_cardinals = 0;

% Final video output size (pixels)
trx3_fig_height = 1080*video_size_coeff;
trx3_fig_width = 1920*video_size_coeff;

%**************************************************************************
% Graphics and overlaid elements
% 3D node radius
node_size_coeff = 0.0025;
r_node_primary = 2.5*node_size_coeff;
r_node_secondary = 3*node_size_coeff;
r_node_tertiary = 1.5*node_size_coeff;
r_node_quaternary = 2*node_size_coeff;
r_node_static = 2*node_size_coeff;

node_surfaces = 20;

edge_width = 4 * video_size_coeff;
edge_sec_width = 1 * video_size_coeff;
edge_sec_color = [.6 .6 .6];
shadow_color_floor = [.6 .6 .6];
shadow_color_wall = [.85 .85 .85];

edge_cage_width = 3.5 * video_size_coeff;
edge_cage_sec_width = 2.5 * video_size_coeff;
edge_cage_sec_color = [.4 .4 .4];

% Reflect axes due to bundle adjustment reflection errors?
reflect_x = 0;
reflect_y = 1;
reflect_z = 1;
%cam_roll = -18;

% Translate and rotate coordinates to center using center of cage?
skeleton_translate = 1;
skeleton_rotate_z_blue = 1;
skeleton_rotate_y_green = 1;
%roll_correction_flip = 0; % set this to 1 if the initial plane is upside down
skeleton_rotate_x_red = 1;

% Plot World Coordinate Axes?
axes_plot_3d = 0;

% Plot reference edges, Cage_median
axes_references_plot_3d = 0;

% Plot surfaces?
plot_faces_cage = 1;
plot_faces_mice = 1;

% Mice colors
%cm = tab10(numAnimals);
cm = [0.254901960784314,0.670588235294118,0.364705882352941;0.258823529411765,0.572549019607843,0.776470588235294;0.866666666666667,0.203921568627451,0.592156862745098];

mouse_alpha = .8;
node_coeff_primary = .3; % Darken the node color to X%
node_coeff_secondary = 1;
node_coeff_tertiary = .6;
node_coeff_quaternary = .8;

node_desaturaion_primary = 0.0; % Desaturate the node color by X%
node_desaturaion_secondary = 0.2;
node_desaturaion_tertiary = 0.5;
node_desaturaion_quaternary = 0.0;

head_color = [.9 .9 .9];
body_color = [.95 .97 1];
ear_color = body_color*.5;
belly_color = [.7 .7 .7];
eye_color = [.4 .4 .4];

% Cage colors
cage_alpha_primary = .3;
cage_color_primary = [0 0 0];
cage_alpha_secondary = .1;
cage_color_secondary = [0 0 0];
cage_color_nodes = [0 0 0];

% Demo pre translate
azs_start = 190;
azs_stop = 190;
els_start = 35;
els_stop = 135;

% Define the 3d space, and how much to move the camera closer
zoom = 1; % 3 means cut the distance between the camera to 1/3 (~3x zoom)
zlim_bot = -1;
zlim_top = 1;
ylim_bot = -1;
ylim_top = 1;
xlim_bot = -1;
xlim_top = 1;


%% Define skeleton
disp(['[' datestr(now, 'HH:MM:SS') '] *Defining skeleton*'])

load(skeleton_file)
numEdges = length(skeleton.edges);
numEdges_sec = length(skeleton.edges_sec);

skeleton_cage = struct();
skeleton_cage.node_names = {...
    'BotNW',...%1
    'BotNE',...%2
    'BotSW',...%3
    'BotSE',...%4
    'BotCenterN',...%5
    'BotCenterS',...%6
    'TopNW',...%7
    'TopNE',...%8
    'TopSW',...%9
    'TopSE',...%10
    'TopCenterN',...%11
    'TopCenterS',...%12
    'SocialPortL',...%13
    'NullPortL',...%14
    'SocialPortR',...%15
    'NullPortR',...%16
    'DoorBotN',...%17
    'DoorTopN',...%18
    'DoorTopS',...%19
    'DoorBotS',...%20
    };
skeleton_cage.node_ind = struct();
for i = 1:numel(skeleton_cage.node_names)
    skeleton_cage.node_ind.(skeleton_cage.node_names{i}) = i;
end

% Define the primary nodes for each skeleton (Nose)
nodes_primary_cage = [1:4,7:10];

% Define the primary edges for the cage
skeleton_cage.edges_primary = [
    skeleton_cage.node_ind.BotNW, skeleton_cage.node_ind.BotNE % 
    skeleton_cage.node_ind.BotNE, skeleton_cage.node_ind.BotSE
    skeleton_cage.node_ind.BotSE, skeleton_cage.node_ind.BotSW
    skeleton_cage.node_ind.BotSW, skeleton_cage.node_ind.BotNW
    skeleton_cage.node_ind.BotNW, skeleton_cage.node_ind.TopNW
    skeleton_cage.node_ind.BotNE, skeleton_cage.node_ind.TopNE
    skeleton_cage.node_ind.BotSE, skeleton_cage.node_ind.TopSE
    skeleton_cage.node_ind.BotSW, skeleton_cage.node_ind.TopSW
    ];
numEdges_cage_primary = size(skeleton_cage.edges_primary,1);

% Define the secondary edges for the cage
skeleton_cage.edges_secondary = [
    skeleton_cage.node_ind.TopNW, skeleton_cage.node_ind.TopNE
    skeleton_cage.node_ind.TopNE, skeleton_cage.node_ind.TopSE
    skeleton_cage.node_ind.TopSE, skeleton_cage.node_ind.TopSW
    skeleton_cage.node_ind.TopSW, skeleton_cage.node_ind.TopNW
    ];
numEdges_cage_secondary = size(skeleton_cage.edges_secondary,1);

% Define the surfaces (faces) for each static
skeleton_cage.walls.floor = [1 2 4 3];
skeleton_cage.walls.North = [1 2 8 7];
skeleton_cage.walls.East = [2 4 10 8];
skeleton_cage.walls.South = [3 4 10 9];
skeleton_cage.walls.West = [3 1 7 9];

faces_cage_primary = [...
    skeleton_cage.walls.floor;...
    ];

faces_cage_secondary = [...
    skeleton_cage.walls.North;...
    skeleton_cage.walls.East;...
    skeleton_cage.walls.South;...
    skeleton_cage.walls.West;...
    ];


%% Define features of the cage to use as coordinate references
% Define cage reference nodes for roll adjustment
Cage_N_ref_roll = 1;
Cage_S_ref_roll = 4;
Cage_W_ref_roll = 2;
Cage_E_ref_roll = 3;

% Define cage reference nodes for height adjustment
cage_height_reference_nodes = [1:4,7:10];

Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));

mean_playpen_edge = mean([...
    pdist([Cage_N;Cage_E]),...
    pdist([Cage_N;Cage_W]),...
    pdist([Cage_S;Cage_W]),...
    pdist([Cage_S;Cage_E])]);

%% Scale coordinates to meters
metric_conversion_coeff = static_reference / mean_playpen_edge;

points_cage = points_cage * metric_conversion_coeff ;
Mice = Mice * metric_conversion_coeff;

% Redefine playpen reference nodes in metric
Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));


%% Define the center of the playpen for translation and roll adjustment
Cage_midfloor = [];
for i=1:3
    Cage_midfloor(i) = mean([Cage_N(i),Cage_E(i),Cage_S(i),Cage_W(i)]);
end


%% Translate skeletons to center of world coordinates relative to cage center
if skeleton_translate == 1

    %points_cage = points_cage - Cage_midfloor;

    for i = 1:size(points_cage,1)
        for j = 1:size(points_cage,2)
            for k = 1:size(points_cage,4)
                points_cage(i,j,1,k) = points_cage(i,j,1,k) - Cage_midfloor(1);
                points_cage(i,j,2,k) = points_cage(i,j,2,k) - Cage_midfloor(2);
                points_cage(i,j,3,k) = points_cage(i,j,3,k) - Cage_midfloor(3);
            end
        end
    end

    for i = 1:size(Mice,1)
        for j = 1:size(Mice,2)
            for k = 1:size(Mice,4)
                Mice(i,j,1,k) = Mice(i,j,1,k) - Cage_midfloor(1);
                Mice(i,j,2,k) = Mice(i,j,2,k) - Cage_midfloor(2);
                Mice(i,j,3,k) = Mice(i,j,3,k) - Cage_midfloor(3);
            end
        end
    end

    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));

end


%% Reflect axes due to bundle adjustment error
% Reflect X axis
if reflect_x == 1
    Mice(:,:,1,:) = Mice(:,:,1,:) * -1;
    points_cage(:,:,1,:) = points_cage(:,:,1,:) * -1;

    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));
end

% Reflect Y axis
if reflect_y == 1
    Mice(:,:,2,:) = Mice(:,:,2,:) * -1;
    points_cage(:,:,2,:) = points_cage(:,:,2,:) * -1;

    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));
end

% Reflect X axis
if reflect_z == 1
    Mice(:,:,3,:) = Mice(:,:,3,:) * -1;
    points_cage(:,:,3,:) = points_cage(:,:,3,:) * -1;

    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));
end


%% Rotate around blue z axis to adjust yaw
z_theta = -1* cart2pol(...
    (Cage_N(1)-Cage_S(1)),(Cage_N(2)-Cage_S(2))...
    ) ;

if skeleton_rotate_z_blue == 1
    % points_cage_temp = points_cage;
    % points_cage(:,1) = points_cage_temp(:,1) * cos(z_theta) - points_cage_temp(:,2) * sin(z_theta);
    % points_cage(:,2) = points_cage_temp(:,1) * sin(z_theta) + points_cage_temp(:,2) * cos(z_theta);

    points_cage_temp = points_cage;
    for i = 1:size(points_cage,1)
        for j = 1:size(points_cage,2)
            for k = 1:size(points_cage,4)
                points_cage(i,j,1,k) = points_cage_temp(i,j,1,k) * cos(z_theta) - points_cage_temp(i,j,2,k)  * sin(z_theta);
                points_cage(i,j,2,k) = points_cage_temp(i,j,1,k) * sin(z_theta) + points_cage_temp(i,j,2,k) * cos(z_theta);
            end
        end
    end
    
    Mice_temp = Mice;
    for i = 1:size(Mice,1)
        for j = 1:size(Mice,2)
            for k = 1:size(Mice,4)
                Mice(i,j,1,k) = Mice_temp(i,j,1,k) * cos(z_theta) - Mice_temp(i,j,2,k)  * sin(z_theta);
                Mice(i,j,2,k) = Mice_temp(i,j,1,k) * sin(z_theta) + Mice_temp(i,j,2,k) * cos(z_theta);
            end
        end
    end
    
    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));
end


%% Rotate around green y axis to adjust pitch

% pitch_rotation_auto = -1 * rad2deg(...
%     cart2pol(...
%     (Cage_N(1)-Cage_S(1)),(Cage_N(3)-Cage_S(3))...
%     ));
% y_theta = pitch_rotation_auto * pi /180;

y_theta = 1 * cart2pol(...
    (Cage_N(1)-Cage_S(1)),(Cage_N(3)-Cage_S(3))...
    );

if skeleton_rotate_y_green == 1
    % points_cage_temp = points_cage;
    % points_cage(:,1) = points_cage_temp(:,1) * cos(y_theta) + points_cage_temp(:,3) * sin(y_theta);
    % points_cage(:,3) = points_cage_temp(:,3) * cos(y_theta) - points_cage_temp(:,1) * sin(y_theta);
    
    points_cage_temp = points_cage;
    for i = 1:size(points_cage,1)
        for j = 1:size(points_cage,2)
            for k = 1:size(points_cage,4)
                points_cage(i,j,1,k) = points_cage_temp(i,j,1,k) * cos(y_theta) + points_cage_temp(i,j,3,k)  * sin(y_theta);
                points_cage(i,j,3,k) = points_cage_temp(i,j,3,k) * cos(y_theta) - points_cage_temp(i,j,1,k) * sin(y_theta);
            end
        end
    end

    Mice_temp = Mice;
    for i = 1:size(Mice,1)
        for j = 1:size(Mice,2)
            for k = 1:size(Mice,4)
                Mice(i,j,1,k) = Mice_temp(i,j,1,k) * cos(y_theta) + Mice_temp(i,j,3,k)  * sin(y_theta);
                Mice(i,j,3,k) = Mice_temp(i,j,3,k) * cos(y_theta) - Mice_temp(i,j,1,k) * sin(y_theta);
            end
        end
    end
        
    % Redefine cage reference nodes
    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));
end


%% Rotate around red x axis to adjust roll

% X-Roll (Y and Z cartesian plane, rotation around Y axis)


% First we need to define a line  for the roll axis 
% roll_rotation_auto = 1 * rad2deg(...
%     cart2pol(...
%     (Cage_E(2)-Cage_W(2)),(Cage_E(3)-Cage_W(3))...
%     ));
% x_theta =  roll_rotation_auto * pi /180;

x_theta =  -1 * cart2pol(...
    (Cage_E(2)-Cage_W(2)),(Cage_E(3)-Cage_W(3))...
    );

if skeleton_rotate_x_red == 1
    % points_cage_temp = points_cage;
    % points_cage(:,2) = points_cage_temp(:,2) * cos(x_theta) - points_cage_temp(:,3) * sin(x_theta);
    % points_cage(:,3) = points_cage_temp(:,2) * sin(x_theta) + points_cage_temp(:,3) * cos(x_theta);

    points_cage_temp = points_cage;
    for i = 1:size(points_cage,1)
        for j = 1:size(points_cage,2)
            for k = 1:size(points_cage,4)
                points_cage(i,j,2,k) = points_cage_temp(i,j,2,k) * cos(x_theta) - points_cage_temp(i,j,3,k) * sin(x_theta);
                points_cage(i,j,3,k) = points_cage_temp(i,j,2,k) * sin(x_theta) + points_cage_temp(i,j,3,k) * cos(x_theta);
            end
        end
    end

    Mice_temp = Mice;
    for i = 1:size(Mice,1)
        for j = 1:size(Mice,2)
            for k = 1:size(Mice,4)
                Mice(i,j,2,k) = Mice_temp(i,j,2,k) * cos(x_theta) - Mice_temp(i,j,3,k) * sin(x_theta);
                Mice(i,j,3,k) = Mice_temp(i,j,2,k) * sin(x_theta) + Mice_temp(i,j,3,k) * cos(x_theta);
            end
        end
    end
    
    % Redefine cage reference nodes
    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));

end

%% Final translation to re-center on the center of the volume rather than the center of the floor

Cage_mid = [];
for j = cage_height_reference_nodes
    for i=1:3
        Cage_mid(i) = mean(squeeze(points_cage(1,j,i))')/2;
    end
end


if skeleton_translate == 1

    %points_cage = points_cage - Cage_midfloor;

    for i = 1:size(points_cage,1)
        for j = 1:size(points_cage,2)
            for k = 1:size(points_cage,4)
                %points_cage(i,j,1,k) = points_cage(i,j,1,k) - Cage_mid(1);
                %points_cage(i,j,2,k) = points_cage(i,j,2,k) - Cage_mid(2);
                points_cage(i,j,3,k) = points_cage(i,j,3,k) - Cage_mid(3);
            end
        end
    end

    for i = 1:size(Mice,1)
        for j = 1:size(Mice,2)
            for k = 1:size(Mice,4)
                %Mice(i,j,1,k) = Mice(i,j,1,k) - Cage_mid(1);
                %Mice(i,j,2,k) = Mice(i,j,2,k) - Cage_mid(2);
                Mice(i,j,3,k) = Mice(i,j,3,k) - Cage_mid(3);
            end
        end
    end

    Cage_N = squeeze(median(points_cage(:,Cage_N_ref_roll,:)));
    Cage_S = squeeze(median(points_cage(:,Cage_S_ref_roll,:)));
    Cage_W = squeeze(median(points_cage(:,Cage_W_ref_roll,:)));
    Cage_E = squeeze(median(points_cage(:,Cage_E_ref_roll,:)));

end

%% Create 3D figure
disp(['[' datestr(now, 'HH:MM:SS') '] *Creating 3D figure*'])

t=v_start;
trx3_fig_fontsize = 18;

vOutput = VideoWriter('/Users/soline/Downloads/trx3.avi',"Uncompressed AVI");
vOutput.FrameRate = FR/FR_subsample;
open(vOutput)

% Create 3D figure
trx3_fig = figure('DefaultAxesFontSize',trx3_fig_fontsize,...
    'Position',  [10, 50, trx3_fig_width, trx3_fig_height]...
    );
%set(gcf, 'Position',  [-2100, 50, trx3_fig_width, trx3_fig_height])
%set(trx3_fig, 'Position',  [10, 50, trx3_fig_width, trx3_fig_height])
trx3_fig.Renderer='Painters';
%figclosekey
%trx3_fig.Renderer='OpenGL';
%trx3_fig.GraphicsSmoothing = 'off';
%trx3_fig.GraphicsSmoothing = 'on';

%set(gca, 'Projection','orthographic')
set(gca,'CameraViewAngleMode','manual')
camproj perspective;
%camproj orthographic;
%campos_default = campos;
%campos(campos_default*zoom)


%ax = gca;
%ax.Clipping = "off";


% Fill plot to entire figure
%InSet = get(gca, 'TightInset');
%set(gca,'position',[0.1 0.1 .8 .8]) % set margins
%set(gca,'position',[-0.375 -0.25 1.75 1.75])
%set(gca,'position',[-.125 -.125 1.25 1.25])
set(gca,'position',[0 0 1 1])
hold on

view(azs_start, els_start)
campos_default = get(gca,'CameraPosition');
campos(campos_default/zoom)

% Plot axes
if axes_plot_3d == 1
    plot3([xlim_bot xlim_top],[0 0],[0 0],'r--','LineWidth',edge_width)
    plot3([0 0],[ylim_bot ylim_top],[0 0],'g--','LineWidth',edge_width)
    plot3([0 0],[0 0],[zlim_bot zlim_top],'b--','LineWidth',edge_width)
end

% Plot reference edges, Cage_median
if axes_references_plot_3d == 1
    % Plot green line between Cage_N and Cage_S
    plot3([Cage_S(1) Cage_N(1)],[Cage_S(2) Cage_N(2)],[Cage_S(3) Cage_N(3)],...
        'k','LineWidth',3,'Color', [.4 .6 .4])
    % Plot purple line between Cage_E and Cage_W
    plot3([Cage_E(1) Cage_W(1)],[Cage_E(2) Cage_W(2)],[Cage_E(3) Cage_W(3)],...
        'k','LineWidth',3,'Color', [.6 .4 .6])
end

% Plot nodes
[sphereX,sphereY,sphereZ] = sphere(node_surfaces); % Build a sphere
hpts = gobjects(numAnimals,numNodes);
for i = 1:numAnimals
    for j = nodes_primary
        hpts(i,j) = surface(...
            sphereX*r_node_primary+Mice(t,j,1,i),sphereY*r_node_primary+Mice(t,j,2,i),sphereZ*r_node_primary+Mice(t,j,3,i),...
            'facecolor',desaturate(cm(i,:),node_desaturaion_primary)*node_coeff_primary,'edgecolor','none');
    end
    for j = nodes_secondary
        hpts(i,j) = surface(...
            sphereX*r_node_secondary+Mice(t,j,1,i),sphereY*r_node_secondary+Mice(t,j,2,i),sphereZ*r_node_secondary+Mice(t,j,3,i),...
            'facecolor',[.2 .2 .2],'edgecolor','none'); %desaturate(cm(i,:),node_desaturaion_secondary)*node_coeff_secondary
    end
    for j = nodes_tertiary
        hpts(i,j) = surface(...
            sphereX*r_node_tertiary+Mice(t,j,1,i),sphereY*r_node_tertiary+Mice(t,j,2,i),sphereZ*r_node_tertiary+Mice(t,j,3,i),...
            'facecolor',desaturate(cm(i,:),node_desaturaion_tertiary)*node_coeff_tertiary,'edgecolor','none');
    end
    % for j = nodes_quaternary
    %     hpts(i,j) = surface(...
    %         sphereX*r_node_quaternary+Mice(t,j,1,i),sphereY*r_node_quaternary+Mice(t,j,2,i),sphereZ*r_node_quaternary+Mice(t,j,3,i),...
    %         'facecolor',desaturate(cm(i,:),node_desaturaion_quaternary)*node_coeff_quaternary,'edgecolor','none');
    % end
end

% Plot nodes of cage
[sphereX_cage,sphereY_cage,sphereZ_cage] = sphere(node_surfaces); % Build a sphere
hpts_cage = gobjects(numStatics,numNodesCage);
for i = 1:numStatics
    for j = nodes_primary_cage
        hpts_cage(i,j) = surface(...
            sphereX_cage*r_node_static+points_cage(1,j,1),sphereY_cage*r_node_static+points_cage(1,j,2),sphereZ_cage*r_node_static+points_cage(1,j,3),...
            'facecolor',cage_color_nodes,'edgecolor','none');
    end
end

% Plot secondary edges, Cage_median
hedges_Cage_median_secondary = gobjects(numStatics,1);
for i = 1:numStatics
    pts_Cage_median_secondary = cell1(numEdges_cage_secondary);
    for k = 1:numEdges_cage_secondary
        src_Cage_median_secondary = squeeze(points_cage(1,skeleton_cage.edges_secondary(k,1),:));
        dst_Cage_median_secondary = squeeze(points_cage(1,skeleton_cage.edges_secondary(k,2),:));
        pts_Cage_median_secondary{k} = [horz(src_Cage_median_secondary); horz(dst_Cage_median_secondary); NaN(1,3)];
    end
    pts_Cage_median_secondary = cellcat(pts_Cage_median_secondary,1);
    hedges_Cage_median_secondary(i) = plot3(pts_Cage_median_secondary(:,1), pts_Cage_median_secondary(:,2), pts_Cage_median_secondary(:,3),...
        '-', 'Color', edge_cage_sec_color, 'LineWidth',edge_cage_sec_width);
end

% Plot primary edges, Cage_median
hedges_Cage_median_primary = gobjects(numStatics,1);
for i = 1:numStatics
    pts_Cage_median_primary = cell1(numEdges_cage_primary);
    for k = 1:numEdges_cage_primary
        src_Cage_median_primary = squeeze(points_cage(1,skeleton_cage.edges_primary(k,1),:));
        dst_Cage_median_primary = squeeze(points_cage(1,skeleton_cage.edges_primary(k,2),:));
        pts_Cage_median_primary{k} = [horz(src_Cage_median_primary); horz(dst_Cage_median_primary); NaN(1,3)];
    end
    pts_Cage_median_primary = cellcat(pts_Cage_median_primary,1);
    hedges_Cage_median_primary(i) = plot3(pts_Cage_median_primary(:,1), pts_Cage_median_primary(:,2), pts_Cage_median_primary(:,3),...
        '-', 'Color', 'k', 'LineWidth',edge_cage_width);
end

% Plot primary edges, mice
hedges = gobjects(numAnimals,1);
for i = 1:numAnimals
    pts = cell1(numEdges);
    for k = 1:numEdges
        src = squeeze(Mice(t,skeleton.edges(k,1),:,i));
        dst = squeeze(Mice(t,skeleton.edges(k,2),:,i));
        pts{k} = [horz(src); horz(dst); NaN(1,3)];
    end
    pts = cellcat(pts,1);
    hedges(i) = plot3(pts(:,1), pts(:,2), pts(:,3),...
        '-', 'Color', cm(i,:), 'LineWidth',edge_width);
end

% Plot secondary edges
hedges_sec = gobjects(numAnimals,1);
for i = 1:numAnimals
    pts = cell1(numEdges_sec);
    for k = 1:numEdges_sec
        src = squeeze(Mice(t,skeleton.edges_sec(k,1),:,i));
        dst = squeeze(Mice(t,skeleton.edges_sec(k,2),:,i));
        pts{k} = [horz(src); horz(dst); NaN(1,3)];
    end
    pts = cellcat(pts,1);
    hedges_sec(i) = plot3(pts(:,1), pts(:,2), pts(:,3),...
        '-', 'Color', edge_sec_color, 'LineWidth',edge_sec_width);
end

% Plot surfaces (faces) of cage skeleton
if plot_faces_cage == 1

    % Build temporary matrix to hold faces of the cage
    faces_primary_temp = nan(size(faces_cage_primary,1)*numStatics,size(faces_cage_primary,2));
    for i=1:numStatics
        faces_primary_temp(size(faces_cage_primary,1)*i - (size(faces_cage_primary,1)-1):size(faces_cage_primary,1)*i,1:size(faces_cage_primary,2)) = ...
            faces_cage_primary+numNodesCage*(i-1);
    end
    faces_secondary_temp = nan(size(faces_cage_secondary,1)*numStatics,size(faces_cage_secondary,2));
    for i=1:numStatics
        faces_secondary_temp(size(faces_cage_secondary,1)*i - (size(faces_cage_secondary,1)-1):size(faces_cage_secondary,1)*i,1:size(faces_cage_secondary,2)) = ...
            faces_cage_secondary+numNodesCage*(i-1);
    end

    % Calculate vertices
    for i = 1:numStatics
        vertices_primary(numNodesCage*(i-1)+1:numNodesCage*(i),1) = squeeze(points_cage(1,:,1));
        vertices_primary(numNodesCage*(i-1)+1:numNodesCage*(i),2) = squeeze(points_cage(1,:,2));
        vertices_primary(numNodesCage*(i-1)+1:numNodesCage*(i),3) = squeeze(points_cage(1,:,3));
    end
    for i = 1:numStatics
        vertices_secondary(numNodesCage*(i-1)+1:numNodesCage*(i),1) = squeeze(points_cage(1,:,1));
        vertices_secondary(numNodesCage*(i-1)+1:numNodesCage*(i),2) = squeeze(points_cage(1,:,2));
        vertices_secondary(numNodesCage*(i-1)+1:numNodesCage*(i),3) = squeeze(points_cage(1,:,3));
    end
    
    % Plot faces
    eval(['faces_M_cage_primary' num2str(i)])= patch(...
        'Faces',faces_primary_temp,'Vertices',vertices_primary,...
        'FaceColor',cage_color_primary,...
        'EdgeColor','none',...
        'FaceAlpha',cage_alpha_primary );
    eval(['faces_M_cage_secondary' num2str(i)])= patch(...
        'Faces',faces_secondary_temp,'Vertices',vertices_secondary,...
        'FaceColor',cage_color_secondary,...
        'EdgeColor','none',...
        'FaceAlpha',cage_alpha_secondary );
   
end

% Plot surfaces (faces) of skeletons
if plot_faces_mice == 1

    % Calculate vertices
    for i = 1:numAnimals
        vertices(numNodes*(i-1)+1:numNodes*(i),1) = squeeze(Mice(t,:,1,i));
        vertices(numNodes*(i-1)+1:numNodes*(i),2) = squeeze(Mice(t,:,2,i));
        vertices(numNodes*(i-1)+1:numNodes*(i),3) = squeeze(Mice(t,:,3,i));
    end
    
    % Build temporary matrix to hold faces of all animals
    faces_temp_body = nan(size(faces_body,1)*numAnimals,size(faces_body,2));
    for i=1:numAnimals
        faces_temp_body(size(faces_body,1)*i - (size(faces_body,1)-1):size(faces_body,1)*i,1:size(faces_body,2)) = ...
            faces_body+numNodes*(i-1);
    end
    
    faces_temp_ears = nan(size(faces_ears,1)*numAnimals,size(faces_ears,2));
    for i=1:numAnimals
        faces_temp_ears(size(faces_ears,1)*i - (size(faces_ears,1)-1):size(faces_ears,1)*i,1:size(faces_ears,2)) = ...
            faces_ears+numNodes*(i-1);
    end
    
    % Plot faces
    eval(['faces_M_body' num2str(i)])= patch(...
    'Faces',faces_temp_body,'Vertices',vertices,...
    'FaceColor',body_color,...
    'EdgeColor','none',...
    'FaceAlpha',mouse_alpha );
    
    eval(['faces_M_ears' num2str(i)])= patch(...
    'Faces',faces_temp_ears,'Vertices',vertices,...
    'FaceColor',ear_color,...
    'EdgeColor','none',...
    'FaceAlpha',mouse_alpha );

    if plot_cardinals == 1
        % Plot text for North and East corners
        text(Cage_N(1),Cage_N(2),Cage_N(3)-.02,'N','Color','red','FontSize',24, 'HorizontalAlignment', 'center')
        text(Cage_W(1),Cage_W(2),Cage_W(3)-.02,'E','Color',[0 .6 .2],'FontSize',24, 'HorizontalAlignment', 'center')
        text(Cage_S(1),Cage_S(2),Cage_S(3)-.02,'S','Color',[0 .6 .2],'FontSize',24, 'HorizontalAlignment', 'center')
        text(Cage_E(1),Cage_E(2),Cage_E(3)-.02,'W','Color',[0 .6 .2],'FontSize',24, 'HorizontalAlignment', 'center')
    end
   
end


ylim([ylim_bot ylim_top])
xlim([xlim_bot xlim_top])
zlim([zlim_bot zlim_top])
%pbaspect([xlim_top-xlim_bot ylim_top-ylim_bot zlim_top-zlim_bot])

xlabel('X (mm)')
ylabel('Y (mm)')
zlabel('Z (mm)')

title('3D projection of matched points using calibrated stereo cameras')
grid on
set(gca,'visible','off')
%set(gca, 'ZDir','reverse')
%set(gca,'visible','on')

%camroll(cam_roll)


%% Animate 3D video
disp(['[' datestr(now, 'HH:MM:SS') '] *Animating 3D video*'])

% vOutput = VideoWriter('/Users/soline/Downloads/trx3.avi',"Uncompressed AVI");
% vOutput.FrameRate = FR/FR_subsample;
% open(vOutput)

azs = linspace(azs_start, azs_stop,v_stop-v_start+1);
els = linspace(els_start, els_stop, v_stop-v_start+1);
mov = cell1(v_stop-v_start+1);

%numFrames
for t = v_start:FR_subsample:v_stop
%for t = 1:1
    disp([ '[' datestr(now, 'HH:MM:SS') '] *Animating figure* (' num2str((t - v_start)/numFrames_output*100,'%02.2f') '%)'])

    %drawnow;
    mov{t} = frame2im(getframe(trx3_fig));

    % Update node positions
    for i = 1:numAnimals
        for j = nodes_primary
            temp_surf = surface(...
                sphereX*r_node_primary+Mice(t,j,1,i),sphereY*r_node_primary+Mice(t,j,2,i),sphereZ*r_node_primary+Mice(t,j,3,i),...
                'facecolor',desaturate(cm(i,:),node_desaturaion_primary)*node_coeff_primary,'edgecolor','none');
            set(hpts(i,j), 'XData', temp_surf.XData, 'YData', temp_surf.YData, 'ZData', temp_surf.ZData)
            delete(temp_surf)
            clear temp_surf
        end
    end
    for i = 1:numAnimals
        for j = nodes_secondary
            temp_surf = surface(...
                sphereX*r_node_secondary+Mice(t,j,1,i),sphereY*r_node_secondary+Mice(t,j,2,i),sphereZ*r_node_secondary+Mice(t,j,3,i),...
                'facecolor',eye_color,'edgecolor','none');
            set(hpts(i,j), 'XData', temp_surf.XData, 'YData', temp_surf.YData, 'ZData', temp_surf.ZData)
            delete(temp_surf)
            clear temp_surf
        end
    end
    for i = 1:numAnimals
        for j = nodes_tertiary
            temp_surf = surface(...
                sphereX*r_node_tertiary+Mice(t,j,1,i),sphereY*r_node_tertiary+Mice(t,j,2,i),sphereZ*r_node_tertiary+Mice(t,j,3,i),...
                'facecolor',desaturate(cm(i,:),node_desaturaion_tertiary)*node_coeff_tertiary,'edgecolor','none');
            set(hpts(i,j), 'XData', temp_surf.XData, 'YData', temp_surf.YData, 'ZData', temp_surf.ZData)
            delete(temp_surf)
            clear temp_surf
        end
    end
    % for i = 1:numAnimals
    %     for j = nodes_quaternary
    %         temp_surf = surface(...
    %             sphereX*r_node_quaternary+Mice(t,j,1,i),sphereY*r_node_quaternary+Mice(t,j,2,i),sphereZ*r_node_quaternary+Mice(t,j,3,i),...
    %             'facecolor',desaturate(cm(i,:),node_desaturaion_quaternary)*node_coeff_quaternary,'edgecolor','none');
    %         set(hpts(i,j), 'XData', temp_surf.XData, 'YData', temp_surf.YData, 'ZData', temp_surf.ZData)
    %         delete(temp_surf)
    %         clear temp_surf
    %     end
    % end
    % 
    % for j = nodes_primary_cage
    %         temp_surf = surface(...
    %             sphereX_cage*r_node_static+points_cage(1,j,1),sphereY_cage*r_node_static+points_cage(1,j,2),sphereZ_cage*r_node_static+points_cage(1,j,3),...
    %         'facecolor',cage_color_primary,'edgecolor','none');
    %         set(hpts_cage(i,j), 'XData', temp_surf.XData, 'YData', temp_surf.YData, 'ZData', temp_surf.ZData)
    %         delete(temp_surf)
    %         clear temp_surf
    % end

    
    % Update primary edge positions
    for i = 1:numAnimals
        pts = cell1(numEdges);
        for k = 1:numEdges
            src = squeeze(Mice(t,skeleton.edges(k,1),:,i));
            dst = squeeze(Mice(t,skeleton.edges(k,2),:,i));
            pts{k} = [horz(src); horz(dst); NaN(1,3)];
        end
        pts = cellcat(pts,1);
        set(hedges(i), 'XData',pts(:,1), 'YData',pts(:,2), 'ZData',pts(:,3))
    end
    
    % Update secondary edge positions
    for i = 1:numAnimals
        pts = cell1(numEdges_sec);
        for k = 1:numEdges_sec
            src = squeeze(Mice(t,skeleton.edges_sec(k,1),:,i));
            dst = squeeze(Mice(t,skeleton.edges_sec(k,2),:,i));
            pts{k} = [horz(src); horz(dst); NaN(1,3)];
        end
        pts = cellcat(pts,1);
        set(hedges_sec(i), 'XData',pts(:,1), 'YData',pts(:,2), 'ZData',pts(:,3))
    end
    
    % Update faces
    if plot_faces_mice == 1
        for i = 1:numAnimals
            vertices(numNodes*(i-1)+1:numNodes*(i),1) = squeeze(Mice(t,:,1,i));
            vertices(numNodes*(i-1)+1:numNodes*(i),2) = squeeze(Mice(t,:,2,i));
            vertices(numNodes*(i-1)+1:numNodes*(i),3) = squeeze(Mice(t,:,3,i));
        end
        set(eval(['faces_M_body' num2str(i)]),...
            'Vertices',vertices);
        set(eval(['faces_M_ears' num2str(i)]),...
            'Vertices',vertices);

    end
    
    % Grab the final frame now that graphics are printed
    currentFrame = getframe(gcf);

    % Write the current frame to memory
    writeVideo(vOutput,currentFrame)

    % Clear the current frame
    %clf
    

end

    
%% Write 3D video to disk
disp(['[' datestr(now, 'HH:MM:SS') '] *Saving 3D video as .mp4*'])

close(vOutput)
%write_frames_ffmpeg(mov,'trx3.avi',FR/FR_subsample)
beep


%% Functions

function a = desaturate(CC,desat)
% Local function that desaturates an rgb color by desat%
    r = CC(1);
    g = CC(2);
    b = CC(3);
    
    L = 0.3*r + 0.6*g + 0.1*b;
    desat_r = r + desat * (L - r);
    desat_g = g + desat * (L - g);
    desat_b = b + desat * (L - b);
    
    a = [desat_r, desat_g, desat_b];
end

function [x y] = GetCircle(r, h, k, a, b)
    t = linspace(a, b, 50);
    x = r*cos(t) + h;
    y = r*sin(t) + k;
end
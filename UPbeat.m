function varargout = UPbeat(varargin)
    % UPBEAT MATLAB code for UPbeat.fig
    %
    %      UPBEAT, by itself, launches the GUI for thermal history
    %      inversion of U-Pb depth profiles.
    %
    %      H = UPBEAT returns the handle to the GUI window.
    %
    %      UPBEAT('CALLBACK',hObject,eventData,handles,...) calls the local
    %      function named CALLBACK in UPBEAT.M with the given input arguments.
    %
    %      UPBEAT('Property','Value',...) creates a new GUI window.
    %      Starting from the left, property value pairs are
    %      applied to the GUI before UPbeat_OpeningFcn gets called.  An
    %      unrecognized property name or invalid value makes property application
    %      stop.  All inputs are passed to UPbeat_OpeningFcn via varargin.

    % Begin initialization code - DO NOT EDIT
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
        'gui_Singleton',  gui_Singleton, ...
        'gui_OpeningFcn', @UPbeat_OpeningFcn, ...
        'gui_OutputFcn',  @UPbeat_OutputFcn, ...
        'gui_LayoutFcn',  [] , ...
        'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end
    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
    % End initialization code - DO NOT EDIT
end

% --- Executes just before UPbeat is made visible.
function UPbeat_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    handles.nnodes = 5;
    handles.Tmax = 825;
    handles.niter = 20000;
    handles.burnin = 20;
    handles.monotonic = true;
    handles.kfiles = {};
    handles.pfiles = {};
    handles.t_nodes = [];
    handles.T_nodes = [];
    handles.LL = [];
    set(handles.nnodebox,'String',handles.nnodes);
    set(handles.TmaxBox,'String',handles.Tmax);
    set(handles.niterbox,'String',handles.niter);
    set(handles.burninbox,'String',handles.burnin);
    set(handles.monotonicradiobutton,'Value',handles.monotonic);
    warning('off','MATLAB:gui:latexsup:UnableToInterpretTeXString');
    % Update handles structure
    guidata(hObject, handles);
end

% --- Outputs from this function are returned to the command line.
function varargout = UPbeat_OutputFcn(hObject, eventdata, handles)
    varargout{1} = handles.t_nodes;
    varargout{2} = handles.T_nodes;
    varargout{3} = handles.LL;
end

function nnodebox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function nnodebox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

function TmaxBox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function TmaxBox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

function niterbox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function niterbox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes on button press in monotonicradiobutton.
function monotonicradiobutton_Callback(hObject, eventdata, handles)
end

function kfilebox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function kfilebox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

function pfilebox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function pfilebox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes on button press in kbutton.
function kbutton_Callback(hObject, eventdata, handles)
    [FileName,PathName,FilterIndex] = uigetfile('*.xls','MultiSelect','on');
    if (isa(FileName,'char'))
        handles.kfiles = cell(1);
        handles.kfiles{1} = strcat(PathName,FileName);
    else
        nf = length(FileName);
        for i = 1:nf,
            handles.kfiles(i) = strcat(PathName,FileName(i));
        end
    end
    set(handles.kfilebox,'String',char(handles.kfiles));
    guidata(hObject, handles);
end

% --- Executes on button press in pbutton.
function pbutton_Callback(hObject, eventdata, handles)
    [FileName,PathName,FilterIndex] = uigetfile('*.txt','MultiSelect','on');
    if (isa(FileName,'char'))
        handles.pfiles = cell(1);
        handles.pfiles{1} = strcat(PathName,FileName);
    else
        nf = length(FileName);
        for i = 1:nf,
            handles.pfiles(i) = strcat(PathName,FileName(i));
        end
    end
    set(handles.pfilebox,'String',char(handles.pfiles));
    guidata(hObject, handles);
end

function burninbox_Callback(hObject, eventdata, handles)
end

% --- Executes during object creation, after setting all properties.
function burninbox_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes on button press in runbutton.
function runbutton_Callback(hObject, eventdata, handles)
    handles.niter = str2double(get(handles.niterbox,'String'));
    handles.nnodes = str2double(get(handles.nnodebox,'String'));
    handles.burnin = str2double(get(handles.burninbox,'String'));
    Tmin = 20; % TODO: ADD BOX FOR MINIMUM TEMPERATURE
    handles.Tmax = str2double(get(handles.TmaxBox,'String'));
    handles.monotonic = get(handles.monotonicradiobutton,'Value');
    num_GridNodes = 512;
    [handles.LL,handles.t_nodes,handles.T_nodes] = ...
        invert(handles.niter,handles.nnodes,handles.kfiles,handles.pfiles, ...
        num_GridNodes,Tmin,handles.Tmax,handles.monotonic);
    guidata(hObject, handles);
    plotter(handles.LL,handles.t_nodes,handles.T_nodes,handles.burnin, ...
        handles.kfiles,handles.pfiles,num_GridNodes);
    % TODO: ADD OPTION TO SAVE OUTPUT AS .CSV FILE
end

%%%%%%%%%%%%%%%%% END OF GUI %%%%%%%%%%%%%%%%%

function [LL,t_nodes,T_nodes] = invert(niter,n_nodes,kineticsfiles,profilefiles,num_GridNodes,min_T,max_T,monotonic)
    nd = length(kineticsfiles);
    M = 0; % initialise maximum apparent age
    SampleData = cell(nd);
    GridNode = cell(nd);
    eff238Conc = cell(nd);
    r_sphere = cell(nd);
    depth = cell(nd);
    mu = cell(nd);
    sigma = cell(nd);
    for i=1:nd,
        disp(strcat('Loading profile #',num2str(i),' ...'));
        [SampleData{i},GridNode{i},eff238Conc{i},r_sphere{i}] = kinetics(kineticsfiles{i},num_GridNodes);
        [depth{i},mu{i},sigma{i}] = profile(profilefiles{i});
        M = max([M,max(mu{i})]);
    end
    t_nodes = [M,linspace(M,0,n_nodes),0]';   % fixed time nodes
    n_walkers = n_nodes*5;
    log_Tn_i = rand(n_nodes,n_walkers);  % initial guess for the log-temperature nodes
    loglik = @(logTn) getLL(logTn,t_nodes,min_T,max_T,monotonic,nd,SampleData,GridNode,...
                         eff238Conc,depth,mu,sigma,r_sphere,num_GridNodes);
    logprior = @(Tn)(prior(Tn));
    %logprior = @(logTn)(true); % flat prior
    [log_Tn,logP] = gwmcmc(log_Tn_i,{logprior loglik},niter);
    %flatten the chain: analyze all the chains as one
    T_nodes = transform(log_Tn,min_T,max_T,monotonic);
    T_nodes = T_nodes(:,:);
    logP = logP(:,:);
    LL = logP(2,:);
end

function valid = prior(uvw)
    valid = all((uvw>-50) & (uvw<50));
end

function [depth,mu,sigma] = profile(datafile)
    profiledat = importdata(datafile);
    depth = profiledat(:,1);
    mu = profiledat(:,2);
    sigma = profiledat(:,3)/2;
end

% maps temperatures from log-space to the [Tmin Tmax] interval
function T = transform(uvw,Tmin,Tmax,monotonic)
    dims = size(uvw);
    nr = dims(1);
    if (length(dims)>2)
        nc = dims(2)*dims(3);
        uvw = reshape(uvw,[nr,nc]);
    else
        nc = dims(2);
    end
    expuvw = [zeros(1,nc);exp(uvw);ones(1,nc)];
    colsum = sum(expuvw,1);
     xyz = expuvw./repmat(colsum,nr+2,1);
    if (monotonic) 
        xyz = ones(nr+2,nc)-cumsum(xyz,1); 
    else
        xyz(end,:) = 0;
    end
    T = Tmin + (Tmax-Tmin) * xyz;
end

% calculate the log-likelihood
function out = getLL(log_T_nodes,t_nodes,min_T,max_T,monotonic,nd,SampleData,...
                     GridNode,eff238Conc,depth,mu,sigma,r_sphere,num_GridNodes)
    T_nodes = transform(log_T_nodes,min_T,max_T,monotonic);
    [t,T] = interpolate(t_nodes,T_nodes,32);
    out = 0;
    for j = 1:nd,
        Conc206_node = Function_UPbDiff(t,T,SampleData{j},GridNode{j},eff238Conc{j});
        out = out + misfit(Conc206_node,eff238Conc{j},depth{j},mu{j},sigma{j},r_sphere{j},num_GridNodes);
    end
end

function out = misfit(Conc206_node,eff238Conc,depth,mu,sigma,r_sphere,num_GridNodes)
    lyr238 = 1.55125e-10;
    Conc206_node_all(:,1) = single(Conc206_node);
    Age206_node_all(:,1)  = log((Conc206_node_all(:,1)./eff238Conc(:))+1)/lyr238;
    rx = linspace(0,r_sphere*10000,num_GridNodes)'; % radial distance in microns
    ry = Age206_node_all/1000000; % modelled ages
    rd = r_sphere*10000-rx; % radial depth
    mage = interp1(rd,ry,depth); % modelled age at the depth of the measurements
    out = - sum( log(sigma) + ((mage-mu).^2)./(2*sigma.^2) );
end

function[t,T] = interpolate(t_nodes,T_nodes,resolution)
    mint = min(t_nodes);
    maxt = max(t_nodes);
    dt = (maxt-mint)/resolution;
    t = [maxt,linspace(maxt-dt/100,mint+dt/100,resolution-2),mint]';
    T = mypchip(t_nodes,T_nodes,t);
end

function T = mypchip(t_nodes,T_nodes,t)
    nr = length(t);
    Tdim = size(T_nodes);
    nc = Tdim(2);
    T = zeros(nr,nc); % initialise
    T(1,:) = T_nodes(1,:);
    T(2:end-1,:) = pchip(t_nodes(2:end-1)',T_nodes(2:end-1,:)',t(2:end-1))';
    T(end,:) = T_nodes(end,:);
end

function [Conc206_node,Conc206_node_t] = Function_UPbDiff(t_nodes,T_nodes,SampleData,GridNode,eff238Conc)

    num_GridNodes = 512;

    % Decay constants
    l238 = 1.55125e-10/365.2425/24/3600;      
    l235 = 9.8485e-10/365.2425/24/3600;

    % INPUT
    Ea  = SampleData.data(9)*1000*0.239;  % convert Activation Energy to cal/mol
    Do  = SampleData.data(10);             % get values from specific location in Sample Data matrix
    dr  = SampleData.data(19);

    t_nodes = double(t_nodes*1000000*365.2425*24*3600); % convert t to seconds as double class
    T_nodes = double(T_nodes+273.15);                   % convert T to Kelvin as double class

    % preallocate Matrices
    rhside_206 = zeros(num_GridNodes-1,1);
    bet_206    = zeros(num_GridNodes-1,1);
    gam_206    = zeros(num_GridNodes-1,1);

    u_206 = zeros(num_GridNodes,1);     % reset Pb206-concentration to 0 before starting the Accumulation/Diffusion calculation  
    for y = 1:size(t_nodes,1)-1         % Loop through time nodes
        dts = t_nodes(y)-t_nodes(y+1);  % calculate delta-t in secs
        Dt  = Do*exp(-Ea/1.9859/((T_nodes(y)+T_nodes(y+1))/2));   % calculate Diffusion coefficient with mean Temperature

        eff206Prod_node = eff238Conc(:).* (exp(l238*t_nodes(y))-exp(l238*t_nodes(y+1)));     % calculate effective Pb206-Production at each node

        % Thomas Algorithm - solving tri-dimensional matrix
        beta_206      = (2*dr^2)/(Dt*dts);
        Term_206      = eff206Prod_node.*GridNode*beta_206;
        mainDiag2_206 = -2-beta_206;
        mainDiag1_206 = mainDiag2_206-1;
        rhside_206(1) = (2-beta_206+1)*u_206(1)-u_206(2)-Term_206(1);
        bet_206(1)    = 1/mainDiag1_206;
        gam_206(1)    = rhside_206(1)/mainDiag1_206;

        for z = 2:num_GridNodes-1       % loop through Grid Nodes  
            rhside_206(z) = -u_206(z-1) + (2-beta_206)*u_206(z) - u_206(z+1) - Term_206(z);
            bet_206(z)    = 1/(mainDiag2_206-bet_206(z-1));
            gam_206(z)    = (rhside_206(z)-gam_206(z-1))/(mainDiag2_206-bet_206(z-1));

        end

        for z = num_GridNodes-1:-1:1   % back-substitution, loop from Grid Node(end-1) to Grid Node(1) 
            u_206(z) = gam_206(z) - bet_206(z)*u_206(z+1);
        end
        u_206_y(:,y) = u_206;   % save u for each time step

        Conc206_node_t(:,y) = u_206./GridNode(:);

    end

    % calculate Pb206-Concentration at each Grid Node at each time step
    % Conc206_node_y  = bsxfun(@rdivide,u_206_y,GridNode(:));
    % Conc206_trpz_y  = (Conc206_node_y(1:end-1,:) + Conc206_node_y(2:end,:))*dr/2;
    % Conc206_y       = sum(bsxfun(@times,Conc206_trpz_y,V_subShell(:)));

    Conc206_node  = u_206./GridNode(:);
    % calculate Pb206-Concentration at each Grid Node
    % Conc206_trpz  = (Conc206_node(1:end-1) + Conc206_node(2:end))*dr/2;
    % Conc206       = sum(Conc206_trpz.* V_subShell(:));

end

function [SampleData,GridNode,eff238Conc,r_sphere] = kinetics(kineticsfile,num_GridNodes)  

    Import_Kinetics = importdata(kineticsfile);  % Import kinetic data
    SampleData(:,:) = Import_Kinetics;

    Geometry      = SampleData.data(1);                  % get values from specific location in Mineral Data matrix
            L             = SampleData.data(2)/10000;    % and convert to cm's
            W             = SampleData.data(3)/10000;
            H             = SampleData.data(4)/10000;
            T             = SampleData.data(5)/10000;
            r             = SampleData.data(6)/10000;
            Density       = SampleData.data(7);
            cU238         = SampleData.data(11);

    %%%%%%%%%%%%%%%%% Equivalent Sphere Radius Calculation %%%%%%%%%%%%%%%%%%%%%%%%%%

            if L*10000 ~= -1 || r*10000 == -1   % use Length and Width for calculations
                switch Geometry
                    case 1 % Spherical Grain
                            r_sphere = (L+W)/2/2;
                            mass     = 4/3*r_sphere^3*pi*Density;
                    case 2 % Elliptical Grain
                            p        = 1.6075;  % parameter p for Ellipsoid Surface calculation
                            Surface  = 4*pi*((H^p*W^p + H^p*L^p + W^p*L^p)/3)^(1/p);
                            Volume   = 4/3*H*W*L*pi;
                            SV_ratio = Surface/Volume;
                            r_sphere = 3/SV_ratio;
                            mass     = Volume*Density;
                    case 3 % Cylindrical Grain
                            Surface  = 2*pi*(W/2)*(W+L);
                            Volume   = (W/2)^2*pi*L;
                            SV_ratio = Surface/Volume;
                            r_sphere = 3/SV_ratio;
                            mass     = Volume*Density;
                    case 4 % Tetragonal prismatic Grain
                            switch T == 0
                                case 1
                                    Surface_B = 2*(W*L) + 2*(H*L) + 2*(H*W);
                                    Volume_B  = H*W*L;
                                    SV_ratio  = Surface_B/Volume_B;
                                    mass = Volume_B*Density;
                                case 0
                                    Surface_B = 2*(W*(L-2*T))+2*(H*(L-2*T));   % no basal planes for Surface area, Tip height substracted from overall Length!!!
                                    Volume_B  = H*W*(L-2*T);
                                    SL_W      = sqrt((H/2)^2 + T^2);     % calculate Side Length for Pyramid face (Base = width)
                                    SL_H      = sqrt((W/2)^2 + T^2);     % calculate Side Length for Pyramid face (Base = height)
                                    Surface_T = 2*(W*SL_W/2 + H*SL_H/2);   % add 2 pairs of Pyramid surfaces  
                                    Volume_T  = 1/3*H*W*T;
                                    SV_ratio  = (2*Surface_T + Surface_B)/(2*Volume_T + Volume_B);
                                    mass  = (Volume_B + 2*Volume_T)*Density;
                            end
                            r_sphere = 3/SV_ratio;

                    case 5 % Hexagonal prismatic Grain
                            Surface  = 2*(3/2*sqrt(3)*(W/2)^2)+6*(W/2*L);
                            Volume   = 3/2*sqrt(3)*(W/2)^2*L; 
                            SV_ratio = Surface/Volume;
                            r_sphere = 3/SV_ratio;
                            mass     = Volume*Density;
                end    
            else                                % use predefined spherical radius for calculations
               r_sphere = r;
               mass     = 4/3*r_sphere^3*pi*Density;           
            end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%% GRID SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dr              = 2*r_sphere/(2*num_GridNodes-1);  % calculate spacing between the Grid Nodes (delta r, Ketcham 2005,p.295)
    GridNodes       = (1:num_GridNodes)';
    GridNode        = dr*GridNodes-dr*0.5;      % defines radial position (cm's from center) of Grid Nodes

    %%%%%%%%%%%%%%%%%%%%%% EFFECTIVE CONCENTRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    eff238Conc = single(1:num_GridNodes);
    eff238Conc(:) = cU238;    % calculate effective U238 Concentration at each node

    SampleData.data(6)  = r_sphere*10000;  % update Sample Data after equivalent sphere calcs
    SampleData.data(8)  = mass;
    SampleData.data(19) = dr;

end

function[map] = diverging_map(s,rgb1,rgb2)
    %This function is based on Kenneth Moreland's code for greating Diverging
    %Colormaps.  Created by Andy Stein.
    %
    %s is a vector that goes between zero and one 
    map = zeros(length(s),3);
    for i=1:length(s)
        map(i,:) = diverging_map_1val(s(i),rgb1,rgb2);
    end
end

% Interpolate a diverging color map.
function[result] = diverging_map_1val(s, rgb1, rgb2)
    %s1 is a number between 0 and 1

    lab1 = RGBToLab(rgb1);
    lab2 = RGBToLab(rgb2);

    msh1 = LabToMsh(lab1);
    msh2 = LabToMsh(lab2);

    % If the endpoints are distinct saturated colors, then place white in between
    % them.
    if msh1(2) > 0.05 && msh2(2) > 0.05 && AngleDiff(msh1(3),msh2(3)) > 0.33*pi    
        % Insert the white midpoint by setting one end to white and adjusting the
        % scalar value.
        Mmid = max(msh1(1), msh2(1));
        Mmid = max(88.0, Mmid);
        if (s < 0.5)
            msh2(1) = Mmid;  msh2(2) = 0.0;  msh2(3) = 0.0;
            s = 2.0*s;
        else
            msh1(1) = Mmid;  msh1(2) = 0.0;  msh1(3) = 0.0;
            s = 2.0*s - 1.0;
        end
    end

    % If one color has no saturation, then its hue value is invalid.  In this
    % case, we want to set it to something logical so that the interpolation of
    % hue makes sense.
    if ((msh1(2) < 0.05) && (msh2(2) > 0.05))
        msh1(3) = AdjustHue(msh2, msh1(1));
    elseif ((msh2(2) < 0.05) && (msh1(2) > 0.05))
        msh2(3) = AdjustHue(msh1, msh2(1));
    end

    mshTmp(1) = (1-s)*msh1(1) + s*msh2(1);
    mshTmp(2) = (1-s)*msh1(2) + s*msh2(2);
    mshTmp(3) = (1-s)*msh1(3) + s*msh2(3);

    % Now convert back to RGB
    labTmp = MshToLab(mshTmp);
    result = LabToRGB(labTmp);
    1;
end

%Convert to and from a special polar version of CIELAB (useful for creating
%continuous diverging color maps).
function[Msh] = LabToMsh(Lab)  
    L = Lab(1);
    a = Lab(2);
    b = Lab(3);

    M = sqrt(L*L + a*a + b*b);
    s = (M > 0.001) * acos(L/M);
    h = (s > 0.001) * atan2(b,a);

    Msh = [M s h];
end

function[Lab] = MshToLab(Msh)
    M = Msh(1);
    s = Msh(2);
    h = Msh(3);

    L = M*cos(s);
    a = M*sin(s)*cos(h);
    b = M*sin(s)*sin(h);

    Lab = [L a b];
end

%Given two angular orientations, returns the smallest angle between the two.
function[adiff] = AngleDiff(a1, a2)
    v1    = [cos(a1) sin(a1)];
    v2    = [cos(a2) sin(a2)];        
    adiff = acos(dot(v1,v2));
end

function[h] = AdjustHue(msh, unsatM)
    if msh(1) >= unsatM-0.1                    
        %The best we can do is hold hue constant.
        h = msh(3);
    else
        % This equation is designed to make the perceptual change of the
        % interpolation to be close to constant.
        hueSpin = (msh(2)*sqrt(unsatM^2 - msh(1)^2)/(msh(1)*sin(msh(2))));

        % Spin hue away from 0 except in purple hues.
        if (msh(3) > -0.3*pi)
            h = msh(3) + hueSpin;
        else
            h = msh(3) - hueSpin;
        end
    end
end

function [xyz] = LabToXYZ(Lab)
    %LAB to XYZ
    L = Lab(1); a = Lab(2); b = Lab(3);

    var_Y = ( L + 16 ) / 116;
    var_X = a / 500 + var_Y;
    var_Z = var_Y - b / 200;

    if ( var_Y^3 > 0.008856 ) 
      var_Y = var_Y^3;
    else
      var_Y = ( var_Y - 16.0 / 116.0 ) / 7.787;
    end
    if ( var_X^3 > 0.008856 ) 
      var_X = var_X^3;
    else
      var_X = ( var_X - 16.0 / 116.0 ) / 7.787;
    end
    if ( var_Z^3) > 0.008856 
      var_Z = var_Z^3;
    else
      var_Z = ( var_Z - 16.0 / 116.0 ) / 7.787;
    end

    ref_X = 0.9505;
    ref_Y = 1.000;
    ref_Z = 1.089;


    x = ref_X * var_X;     %ref_X = 0.9505  Observer= 2 deg Illuminant= D65
    y = ref_Y * var_Y;     %ref_Y = 1.000
    z = ref_Z * var_Z;     %ref_Z = 1.089

    xyz = [x y z];
end

function[Lab] = XYZToLab(xyz)
    x = xyz(1); y = xyz(2); z = xyz(3);

    ref_X = 0.9505;
    ref_Y = 1.000;
    ref_Z = 1.089;
    var_X = x / ref_X;  %ref_X = 0.9505  Observer= 2 deg, Illuminant= D65
    var_Y = y / ref_Y;  %ref_Y = 1.000
    var_Z = z / ref_Z;  %ref_Z = 1.089

    if ( var_X > 0.008856 ), var_X = var_X^(1/3);
    else                     var_X = ( 7.787 * var_X ) + ( 16.0 / 116.0 ); end
    if ( var_Y > 0.008856 ), var_Y = var_Y^(1/3);
    else                     var_Y = ( 7.787 * var_Y ) + ( 16.0 / 116.0 ); end
    if ( var_Z > 0.008856 ), var_Z = var_Z^(1/3);
    else                     var_Z = ( 7.787 * var_Z ) + ( 16.0 / 116.0 ); end

    L = ( 116 * var_Y ) - 16;
    a = 500 * ( var_X - var_Y );
    b = 200 * ( var_Y - var_Z );

    Lab = [L a b];
end

function[rgb] = XYZToRGB(xyz)  
  %ref_X = 0.9505;        %Observer = 2 deg Illuminant = D65
  %ref_Y = 1.000;
  %ref_Z = 1.089;
  
  x = xyz(1); y = xyz(2); z = xyz(3);
  r = x *  3.2406 + y * -1.5372 + z * -0.4986;
  g = x * -0.9689 + y *  1.8758 + z *  0.0415;
  b = x *  0.0557 + y * -0.2040 + z *  1.0570;

  % The following performs a "gamma correction" specified by the sRGB color
  % space.  sRGB is defined by a canonical definition of a display monitor and
  % has been standardized by the International Electrotechnical Commission (IEC
  % 61966-2-1).  The nonlinearity of the correction is designed to make the
  % colors more perceptually uniform.  This color space has been adopted by
  % several applications including Adobe Photoshop and Microsoft Windows color
  % management.  OpenGL is agnostic on its RGB color space, but it is reasonable
  % to assume it is close to this one.
  if (r > 0.0031308), r = 1.055 * r^( 1 / 2.4 ) - 0.055;
  else r = 12.92 * (r); end
  if (g > 0.0031308), g = 1.055 * g^( 1 / 2.4 ) - 0.055;
  else  g = 12.92 * (g); end
  if (b > 0.0031308), b = 1.055 * b^( 1 / 2.4 ) - 0.055;
  else b = 12.92 * (b); end

  % Clip colors. ideally we would do something that is perceptually closest
  % (since we can see colors outside of the display gamut), but this seems to
  % work well enough.
  maxVal = r;
  if (maxVal < g), maxVal = g; end
  if (maxVal < b), maxVal = b; end
  if (maxVal > 1.0)    
    r = r/maxVal;
    g = g/maxVal;
    b = b/maxVal;
  end
  if (r<0), r=0; end
  if (g<0), g=0; end
  if (b<0), b=0; end
  
  rgb = [r g b];
end

%-----------------------------------------------------------------------------
function[xyz] = RGBToXYZ(rgb)
  r = rgb(1); g = rgb(2); b = rgb(3);

  % The following performs a "gamma correction" specified by the sRGB color
  % space.  sRGB is defined by a canonical definition of a display monitor and
  % has been standardized by the International Electrotechnical Commission (IEC
  % 61966-2-1).  The nonlinearity of the correction is designed to make the
  % colors more perceptually uniform.  This color space has been adopted by
  % several applications including Adobe Photoshop and Microsoft Windows color
  % management.  OpenGL is agnostic on its RGB color space, but it is reasonable
  % to assume it is close to this one.
  if ( r > 0.04045 ), r = (( r + 0.055 ) / 1.055)^2.4;
  else                r = r / 12.92; end
  if ( g > 0.04045 ), g = (( g + 0.055 ) / 1.055)^2.4;
  else                g = g / 12.92; end
  if ( b > 0.04045 ), b = (( b + 0.055 ) / 1.055)^2.4;
  else                b = b / 12.92; end

  %Observer. = 2 deg, Illuminant = D65
  x = r * 0.4124 + g * 0.3576 + b * 0.1805;
  y = r * 0.2126 + g * 0.7152 + b * 0.0722;
  z = r * 0.0193 + g * 0.1192 + b * 0.9505;
  
  xyz = [x y z];
end

function[rgb] = LabToRGB(Lab)
  xyz = LabToXYZ(Lab);
  rgb = XYZToRGB(xyz);
end

function[Lab] = RGBToLab(rgb)
  xyz = RGBToXYZ(rgb);
  Lab = XYZToLab(xyz);
end


% Plot radial age profile
function plotoutput(t_nodes,T_nodes,kineticsfiles,profilefiles,num_GridNodes)
    lyr238 = 1.55125e-10;
    nd = length(kineticsfiles); % number of datasets
    np = ceil(sqrt(nd)); % number of plots
    for i = 1:nd,
        [SampleData{i},GridNode{i},eff238Conc{i},r_sphere{i}] = kinetics(kineticsfiles{i},num_GridNodes);
        [depth{i},mu{i},sigma{i}] = profile(profilefiles{i});
    end
    [t,T] = interpolate(t_nodes,T_nodes,32);
    figure
    for i = 1:nd,
        Conc206_node = Function_UPbDiff(t,T,SampleData{i},GridNode{i},eff238Conc{i});
        Conc206_node_all(:,1) = single(Conc206_node);
        Age206_node_all(:,1)  = log((Conc206_node_all(:,1)./eff238Conc{i}(:))+1)/lyr238;
        rx = linspace(0,r_sphere{i}*10000,num_GridNodes); % radial distance in microns
        ry = Age206_node_all/1000000; % modelled ages
        subplot(np,np,i)
        plot(rx,ry,'-b'); hold on;
        errorbar(r_sphere{i}*10000-depth{i},mu{i},2*sigma{i},'.k');
        xlabel('radial distance'); ylabel('age'); title(profilefiles{i});
    end
    %figure;
    %plot(t,T); xlabel('time'); ylabel('Temperature')
end

function plotter(LL,t_nodes,T_nodes,burnin,kineticsfiles,profilefiles,num_GridNodes)

    close all;

    n  = length(LL);
    b = round(length(LL)*burnin/100);
    [ML,mi] = max(LL);
    best_T = T_nodes(:,mi);

    resolution = 32;

    % plotting
    % fig 1: LL with time
    figure
    disp('Plotting the Likelihood ...');
    plot(1:length(LL)-1,LL(2:end),'-k'); hold on;
    plot([b,b],ylim,'-b');
    xlabel('iteration'); ylabel('log-likelihood');

    % fig 2: Tt paths shaded for LL
    figure
    disp('Plotting the t-T paths ...');
    hold on
    s = linspace(0,1,100);
    rgb1 = [0.230, 0.299, 0.754];
    rgb2 = [0.706, 0.016, 0.150];
    cmap = diverging_map(s,rgb1,rgb2);
    t = interpolate(t_nodes,T_nodes,resolution);
    T = zeros(resolution,n-b);
    mL = min(LL(b:n));
    for i=1:n-b,
        T(:,i) = mypchip(t_nodes,T_nodes(:,i+b-1),t);
        ci = floor(99*(LL(i+b-1)-mL)/(ML-mL))+1; % colour index
        plot(t, T(:,i), 'Color', [cmap(ci,1) cmap(ci,2) cmap(ci,3)]);
    end
    colormap(cmap);
    c=colorbar;
    caxis([min(LL) max(LL)]);
    set(get(c,'ylabel'),'string','Log Likelihood','fontsize',16);
    set(gca,'Xdir','reverse','Ydir','reverse');
    xlabel('Time (Ma)','fontsize',16); ylabel('Temperature (^{\circ}C)','fontsize',16);
    TT = mypchip(t_nodes,best_T,t);
    plot(t(2:end-1),TT(2:end-1),'k','LineWidth',2); box on; grid on;
    for i = 1:length(t_nodes),
        plot([t_nodes(i),t_nodes(i)],ylim,':k');
    end
    hold off
    
    disp('Plotting the predicted and observed data ...');
    % fig 3: U-Pb age profiles against data
    plotoutput(t_nodes,best_T,kineticsfiles,profilefiles,num_GridNodes);

end

function [models,logP]=gwmcmc(minit,logPfuns,mccount,varargin)
    % Cascaded affine invariant ensemble MCMC sampler. "The MCMC hammer"
    %
    % GWMCMC is an implementation of the Goodman and Weare 2010 Affine
    % invariant ensemble Markov Chain Monte Carlo (MCMC) sampler. MCMC sampling
    % enables bayesian inference. The problem with many traditional MCMC samplers
    % is that they can have slow convergence for badly scaled problems, and that
    % it is difficult to optimize the random walk for high-dimensional problems.
    % This is where the GW-algorithm really excels as it is affine invariant. It
    % can achieve much better convergence on badly scaled problems. It is much
    % simpler to get to work straight out of the box, and for that reason it
    % truly deserves to be called the MCMC hammer.
    %
    % (This code uses a cascaded variant of the Goodman and Weare algorithm).
    %
    % USAGE:
    %  [models,logP]=gwmcmc(minit,logPfuns,mccount,[Parameter,Value,Parameter,Value]);
    %
    % INPUTS:
    %     minit: an MxW matrix of initial values for each of the walkers in the
    %            ensemble. (M:number of model params. W: number of walkers). W
    %            should be atleast 2xM. (see e.g. mvnrnd).
    %  logPfuns: a cell of function handles returning the log probality of a
    %            proposed set of model parameters. Typically this cell will
    %            contain two function handles: one to the logprior and another
    %            to the loglikelihood. E.g. {@(m)logprior(m) @(m)loglike(m)}
    %   mccount: What is the desired total number of monte carlo proposals.
    %            This is the total number, -NOT the number per chain.
    %
    % Named Parameter-Value pairs:
    %   'StepSize': unit-less stepsize (default=2.5).
    %   'ThinChain': Thin all the chains by only storing every N'th step (default=10)
    %   'ProgressBar': Show a text progress bar (default=true)
    %   'Parallel': Run in ensemble of walkers in parallel. (default=false)
    %   'BurnIn': fraction of the chain that should be removed. (default=0)
    %
    % OUTPUTS:
    %    models: A MxWxT matrix with the thinned markov chains (with T samples
    %            per walker). T=~mccount/p.ThinChain/W.
    %      logP: A PxWxT matrix of log probabilities for each model in the
    %            models. here P is the number of functions in logPfuns.
    %
    % Note on cascaded evaluation of log probabilities:
    % The logPfuns-argument can be specifed as a cell-array to allow a cascaded
    % evaluation of the probabilities. The computationally cheapest function should be
    % placed first in the cell (this will typically the prior). This allows the
    % routine to avoid calculating the likelihood, if the proposed model can be
    % rejected based on the prior alone.
    % logPfuns={logprior loglike} is faster but equivalent to
    % logPfuns={@(m)logprior(m)+loglike(m)}
    %
    % TIP: if you aim to analyze the entire set of ensemble members as a single
    % sample from the distribution then you may collapse output models-matrix
    % thus: models=models(:,:); This will reshape the MxWxT matrix into a
    % Mx(W*T)-matrix while preserving the order.
    %
    %
    % EXAMPLE: Here we sample a multivariate normal distribution.
    %
    % %define problem:
    % mu = [5;-3;6];
    % C = [.5 -.4 0;-.4 .5 0; 0 0 1];
    % iC=pinv(C);
    % logPfuns={@(m)-0.5*sum((m-mu)'*iC*(m-mu))}
    %
    % %make a set of starting points for the entire ensemble of walkers
    % minit=randn(length(mu),length(mu)*2);
    %
    % %Apply the MCMC hammer
    % [models,logP]=gwmcmc(minit,logPfuns,100000);
    % models(:,:,1:floor(size(models,3)*.2))=[]; %remove 20% as burn-in
    % models=models(:,:)'; %reshape matrix to collapse the ensemble member dimension
    % scatter(models(:,1),models(:,2))
    % prctile(models,[5 50 95])
    %
    %
    % References:
    % Goodman & Weare (2010), Ensemble Samplers With Affine Invariance, Comm. App. Math. Comp. Sci., Vol. 5, No. 1, 65�80
    % Foreman-Mackey, Hogg, Lang, Goodman (2013), emcee: The MCMC Hammer, arXiv:1202.3665
    %
    % WebPage: https://github.com/grinsted/gwmcmc
    %
    % -Aslak Grinsted 2015

    persistent isoctave;
    if isempty(isoctave)
        isoctave = (exist ('OCTAVE_VERSION', 'builtin') > 0);
    end

    if nargin<3
        error('GWMCMC:toofewinputs','GWMCMC requires atleast 3 inputs.')
    end
    M=size(minit,1);
    if size(minit,2)==1
        minit=bsxfun(@plus,minit,randn(M,M*5));
    end

    p=inputParser;
    if isoctave
        p=p.addParamValue('StepSize',2,@isnumeric); %addParamValue is chose for compatibility with octave. Still Untested.
        p=p.addParamValue('ThinChain',10,@isnumeric);
        p=p.addParamValue('ProgressBar',true,@islogical);
        p=p.addParamValue('Parallel',false,@islogical);
        p=p.addParamValue('BurnIn',0,@(x)(x>=0)&&(x<1));
        p=p.parse(varargin{:});
    else
        p.addParamValue('StepSize',2,@isnumeric); %addParamValue is chose for compatibility with octave. Still Untested.
        p.addParamValue('ThinChain',10,@isnumeric);
        p.addParamValue('ProgressBar',true,@islogical);
        p.addParamValue('Parallel',false,@islogical);
        p.addParamValue('BurnIn',0,@(x)(x>=0)&&(x<1));
        p.parse(varargin{:});
    end
    p=p.Results;

    Nwalkers=size(minit,2);

    if size(minit,1)*2>size(minit,2)
        warning('GWMCMC:minitdimensions','Check minit dimensions.\nIt is recommended that there be atleast twice as many walkers in the ensemble as there are model dimension.')
    end

    if p.ProgressBar
        progress=@textprogress;
    else
        progress=@noaction;
    end

    Nkeep=ceil(mccount/p.ThinChain/Nwalkers); %number of samples drawn from each walker
    mccount=(Nkeep-1)*p.ThinChain+1;

    models=nan(M,Nwalkers,Nkeep); %pre-allocate output matrix

    models(:,:,1)=minit;

    if ~iscell(logPfuns)
        logPfuns={logPfuns};
    end

    NPfun=numel(logPfuns);

    %calculate logP state initial pos of walkers
    logP=nan(NPfun,Nwalkers,Nkeep);
    for wix=1:Nwalkers
        for fix=1:NPfun
            v=logPfuns{fix}(minit(:,wix));
            if islogical(v) %reformulate function so that false=-inf for logical constraints.
                v=-1/v;logPfuns{fix}=@(m)-1/logPfuns{fix}(m); %experimental implementation of experimental feature
            end
            logP(fix,wix,1)=v;
        end
    end

    if ~all(all(isfinite(logP(:,:,1))))
        error('Starting points for all walkers must have finite logP')
    end

    reject=zeros(Nwalkers,1);

    curm=models(:,:,1);
    curlogP=logP(:,:,1);
    progress(0,0,0)
    totcount=Nwalkers;
    for row=1:Nkeep
        for jj=1:p.ThinChain
            %generate proposals for all walkers
            %(done outside walker loop, in order to be compatible with parfor - some penalty for memory):
            %-Note it appears to give a slight performance boost for non-parallel.
            rix=mod((1:Nwalkers)+floor(rand*(Nwalkers-1)),Nwalkers)+1; %pick a random partner
            zz=((p.StepSize - 1)*rand(1,Nwalkers) + 1).^2/p.StepSize;
            proposedm=curm(:,rix) - bsxfun(@times,(curm(:,rix)-curm),zz);
            logrand=log(rand(NPfun+1,Nwalkers)); %moved outside because rand is slow inside parfor
            if p.Parallel
                %parallel/non-parallel code is currently mirrored in
                %order to enable experimentation with separate optimization
                %techniques for each branch. Parallel is not really great yet.
                %TODO: use SPMD instead of parfor.

                parfor wix=1:Nwalkers
                    cp=curlogP(:,wix);
                    lr=logrand(:,wix);
                    acceptfullstep=true;
                    proposedlogP=nan(NPfun,1);
                    if lr(1)<(numel(proposedm(:,wix))-1)*log(zz(wix))
                        for fix=1:NPfun
                            proposedlogP(fix)=logPfuns{fix}(proposedm(:,wix)); %have tested workerobjwrapper but that is slower.
                            if lr(fix+1)>proposedlogP(fix)-cp(fix) || ~isreal(proposedlogP(fix)) || isnan( proposedlogP(fix) )
                                %if ~(lr(fix+1)<proposedlogP(fix)-cp(fix))
                                acceptfullstep=false;
                                break
                            end
                        end
                    else
                        acceptfullstep=false;
                    end
                    if acceptfullstep
                        curm(:,wix)=proposedm(:,wix); curlogP(:,wix)=proposedlogP;
                    else
                        reject(wix)=reject(wix)+1;
                    end
                end
            else %NON-PARALLEL
                for wix=1:Nwalkers
                    acceptfullstep=true;
                    proposedlogP=nan(NPfun,1);
                    if logrand(1,wix)<(numel(proposedm(:,wix))-1)*log(zz(wix))
                        for fix=1:NPfun
                            proposedlogP(fix)=logPfuns{fix}(proposedm(:,wix));
                            if logrand(fix+1,wix)>proposedlogP(fix)-curlogP(fix,wix) || ~isreal(proposedlogP(fix)) || isnan(proposedlogP(fix))
                                %if ~(logrand(fix+1,wix)<proposedlogP(fix)-curlogP(fix,wix)) %inverted expression to ensure rejection of nan and imaginary logP's.
                                acceptfullstep=false;
                                break
                            end
                        end
                    else
                        acceptfullstep=false;
                    end
                    if acceptfullstep
                        curm(:,wix)=proposedm(:,wix); curlogP(:,wix)=proposedlogP;
                    else
                        reject(wix)=reject(wix)+1;
                    end
                end

            end
            totcount=totcount+Nwalkers;
            progress((row-1+jj/p.ThinChain)/Nkeep,curm,sum(reject)/totcount)
        end
        models(:,:,row)=curm;
        logP(:,:,row)=curlogP;

        %progress bar

    end
    progress(1,0,0);
    if p.BurnIn>0
        crop=ceil(Nkeep*p.BurnIn);
        models(:,:,1:crop)=[]; %TODO: never allocate space for them ?
        logP(:,:,1:crop)=[];
    end
end

function textprogress(pct,curm,rejectpct)
    persistent lastNchar lasttime starttime
    if isempty(lastNchar)||pct==0
        lasttime=cputime-10;starttime=cputime;lastNchar=0;
        pct=1e-16;
    end
    if pct==1
        fprintf('%s',repmat(char(8),1,lastNchar));lastNchar=0;
        return
    end
    if (cputime-lasttime>0.1)

        ETA=datestr((cputime-starttime)*(1-pct)/(pct*60*60*24),13);
        progressmsg=[183-uint8((1:40)<=(pct*40)).*(183-'*') ''];
        curmtxt=sprintf('% 9.3g\n',curm(1:min(end,20),1));
        progressmsg=sprintf('\nGWMCMC %5.1f%% [%s] %s\n%3.0f%% rejected\n%s\n',pct*100,progressmsg,ETA,rejectpct*100,curmtxt);

        fprintf('%s%s',repmat(char(8),1,lastNchar),progressmsg);
        drawnow;lasttime=cputime;
        lastNchar=length(progressmsg);
    end
end

function noaction(varargin)
end

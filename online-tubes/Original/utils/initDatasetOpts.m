% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function opts = initDatasetOpts(data_root,baseDir,dataset,imgType,model_type,listid,iteration_num,iouthresh,costtype,gap, exp_id)

opts = struct();
opts.imgType = imgType;             % imgType :: rgb | brox | fastOF
opts.costtype = costtype;           % costtype :: how the similarity score is calculated 'score'
opts.gap = gap;                     % jump gap :: number of frames to look for continuation
opts.baseDir = baseDir;             % save dir path
opts.dataset = dataset;             % dataset :: ucf24 | jhmdb-21
opts.iouThresh = iouthresh;         % iouThresh
opts.weight = iteration_num;        % iteration_num
opts.listid = listid;               % split file no :: 01

testlist = ['testlist',listid];
% testlist = 'testlist_';
%opts.vidList = sprintf('../../test_tmap/splitfiles/%s.txt',testlist);
opts.vidList = sprintf('%s/%s/splitfiles/%s.txt',data_root, dataset, testlist)

if strcmp(dataset,'ucf24')
    opts.actions = {'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',...
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',...
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',...
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',...
        'VolleyballSpiking','WalkingWithDog'};
elseif strcmp(dataset,'hmdb21')
    opts.actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
        'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
        'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};
elseif strcmp(dataset,'LIRIS')
    opts.actions = {'discussion', 'give_object_to_person','put_take_obj_into_from_box_desk',...
        'enter_leave_room_no_unlocking','try_enter_room_unsuccessfully','unlock_enter_leave_room',...
        'leave_baggage_unattended','handshaking','typing_on_keyboard','telephone_conversation'};
end

opts.imgDir = sprintf('%s/%s/%s-images/',data_root,dataset,imgType);

opts.detDir = sprintf('../../RealTime/%s/results/%s_%s_%s_100dets/',model_type, dataset, imgType, exp_id);

opts.annotFile = sprintf('%s/%s/splitfiles/finalAnnots.ma.mat',data_root,dataset);

opts.actPathDir = sprintf('%s/%s/%s_%s/actionPaths/',baseDir,dataset,imgType, exp_id);    % SHOULD CHANGE
opts.tubeDir = sprintf('%s/%s/%s_%s/actionTubes/',baseDir,dataset,imgType, exp_id);       % SHOULD CHANGE

if exist(opts.detDir,'dir')
    if ~isdir(opts.actPathDir)
        fprintf('Creating %s\n',opts.actPathDir);
        mkdir(opts.actPathDir)
    end
    if ~isdir(opts.tubeDir)
        mkdir(opts.tubeDir)
    end
    if strcmp(dataset,'ucf24') || strcmp(dataset,'hmdb21')
        createdires({opts.actPathDir},opts.actions)
    end
end

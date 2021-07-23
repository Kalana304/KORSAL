% ---------------------------------------------------------
% Copyright (c) 2017, Gurkirt Singh
% This code and is available
% under the terms of MIT License provided in LICENSE.
% Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

function opts = initDatasetOptsFused(data_root,baseDir,dataset,imtypes,model_type, ...
    listid,iteration_nums,iouthresh,costtype,gap,fusiontype,fuseiouth)
%% data_root,baseDir,dataset,imgType,model_type,listid,iteration_num,iouthresh,costtype,gap

opts = struct();
imgType = [imtypes{1},'-',imtypes{2}];
opts.imgType = imgType;
opts.costtype = costtype;
opts.gap = gap;
opts.baseDir = baseDir;
opts.imgType = imgType;
opts.dataset = dataset;
opts.iouThresh = iouthresh;
opts.iteration_nums = iteration_nums;
opts.listid = listid;
opts.fusiontype = fusiontype;
opts.fuseiouth = fuseiouth;
testlist = ['testlist',listid];
opts.data_root = data_root;
opts.vidList = sprintf('%s/%s/splitfiles/%s.txt',data_root,dataset,testlist);

if strcmp(dataset,'ucf24')
    opts.actions = {'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',...
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',...
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',...
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',...
        'VolleyballSpiking','WalkingWithDog'};
elseif strcmp(dataset,'JHMDB')
    opts.actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
        'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
        'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};
elseif strcmp(dataset,'LIRIS')
    opts.actions = {'discussion', 'give_object_to_person','put_take_obj_into_from_box_desk',...
        'enter_leave_room_no_unlocking','try_enter_room_unsuccessfully','unlock_enter_leave_room',...
        'leave_baggage_unattended','handshaking','typing_on_keyboard','telephone_conversation'};
end

opts.imgDir = sprintf('%s/%s/%s-images/',data_root,dataset,imtypes{1});

opts.basedetDir = sprintf('%s/%s/detections/%s-%s-%s-%06d/',baseDir,dataset,model_type,imtypes{1},listid,iteration_nums(1));
opts.topdetDir = sprintf('%s/%s/detections/%s-%s-%s-%06d/',baseDir,dataset,model_type,imtypes{2},listid,iteration_nums(2));

opts.annotFile = sprintf('%s/%s/splitfiles/annots.mat',data_root,dataset);

opts.actPathDir = sprintf('%s/%s/actionPaths/%s/%s-%s-%s-%s-%d-%d-%s-%d-%04d-fiou%03d/',baseDir,dataset,fusiontype,model_type,imtypes{1},imtypes{2},...
                                        listid,iteration_nums(1),iteration_nums(2),costtype,gap,iouthresh*100,uint16(fuseiouth*100));
opts.tubeDir = sprintf('%s/%s/actionTubes/%s/%s-%s-%s-%s-%d-%d-%s-%d-%04d-fiou%03d/',baseDir,dataset,fusiontype,model_type,imtypes{1},imtypes{2},...
                                        listid,iteration_nums(1),iteration_nums(2),costtype,gap,iouthresh*100,uint16(fuseiouth*100));

if exist(opts.basedetDir,'dir')
    if ~isdir(opts.actPathDir)
        fprintf('Creating %s\n',opts.actPathDir);
        mkdir(opts.actPathDir)
    end
    
    if ~isdir(opts.tubeDir)
        mkdir(opts.tubeDir)
    end
    
    if strcmp(dataset,'ucf24') || strcmp(dataset,'JHMDB')
        createdires({opts.actPathDir},opts.actions)
    end
end

%fprintf('Video List :: %s\nImage  Dir :: %s\nDetection Dir:: %s\nActionpath Dir:: %s\nTube Dir:: %s\n',...
 %    opts.vidList,opts.imgDir,opts.detDir,opts.actPathDir,opts.tubeDir)

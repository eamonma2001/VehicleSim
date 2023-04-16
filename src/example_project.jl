include("./tools/astar.jl")

struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
    v::Float64
    l::Float64
    w::Float64
    h::Float64
end

struct FullVehicleState
    position::SVector{3, Float64}
    velocity::SVector{3, Float64}
    orientation::SVector{3, Float64}
    angular_vel::SVector{3, Float64}
end

struct MyLocalizationType
    last_update::Float64
    x::FullVehicleState
end

struct MyPerceptionType
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
        fresh_gps_meas = []
        while isready(gps_channel)
            meas = take!(gps_channel)
            push!(fresh_gps_meas, meas)
        end
        fresh_imu_meas = []
        while isready(imu_channel)
            meas = take!(imu_channel)
            push!(fresh_imu_meas, meas)
        end
        
        # process measurements

        localization_state = MyLocalizationType(0,0.0)
        if isready(localization_state_channel)
            take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end 
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    while true
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end

        latest_localization_state = fetch(localization_state_channel)
        
        # process bounding boxes / run ekf / do what you think is good

        perception_state = MyPerceptionType(0,0.0)
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

function if_in_segments(seg, ego_location)
    lb_1 = seg.lane_boundaries[1]
    if length(seg.lane_boundaries) == 2
        lb_2 = seg.lane_boundaries[2]
    else
        lb_2 = seg.lane_boundaries[3]
    end

    pt_a = lb_1.pt_a
    pt_b = lb_1.pt_b
    pt_c = lb_2.pt_a
    pt_d = lb_2.pt_b

    curvature = lb_1.curvature
    curved = !isapprox(curvature, 0.0; atol=1e-6)
    delta = pt_b-pt_a
    delta2 = pt_d-pt_b
    if !curved
        pt = 0.25*(pt_a+pt_b+pt_c+pt_d)
        check = abs(pt[1] - ego_location[1])
        check2 = abs(pt[2] - ego_location[2])
        if delta[1] == 0    
            if check < abs(delta2[1]/2) 
                if check2 < abs(delta[2]/2) 
                    return true
                else
                    return false
                end
            else
                return false
            end
        elseif delta[2] == 0
            if check < abs(delta[1]/2) 
                if check2 < abs(delta2[2]/2) 
                    return true
                else
                    return false
                end
            else
                return false
            end
        end
    else
        rad = 1.0 / abs(curvature)
        dist = π*rad/2.0
        left = curvature > 0

        rad_1 = rad
        #@info "rad_1: $rad_1"
        rad_2 = abs(pt_d[1]-pt_c[1])
        #@info "rad_2: $rad_2"

        if left
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [0, delta[2]]
            else
                center = pt_a + [delta[1], 0]
            end
        else
            if sign(delta[1]) == sign(delta[2])
                center = pt_a + [delta[1], 0]
            else
                center = pt_a + [0, delta[2]]
            end
        end

        r = (ego_location[1:2]-center[1:2])'*(ego_location[1:2]-center[1:2])
        #@info "center: $center, r: $r"
        if r < rad_1*rad_1
            return false
        end
        if r > rad_2*rad_2
            return false
        end
        if r > rad_1*rad_1
            if rad_2*rad_2 < r
                if left
                    if sign(delta[1]) == sign(delta[2])
                        if ego_location[1] < center[1] 
                            if ego_location[2] > center[2]
                                return true
                            end
                        end
                    end
                else
                    if ego_location[1] < center[1] 
                        if ego_location[2] < center[2]
                            return true
                        end
                    end
                end
            else
                if sign(delta[1]) == sign(delta[2])
                    if ego_location[1] < center[1] 
                        if ego_location[2] > center[2]
                            return true
                        end
                    end
                else
                    if ego_location[1] > center[1]
                        if ego_location[2] > center[2]
                            return true
                        end
                    end
                end
            end
        end
    end
    return false
end

function decision_making(localization_state_channel, 
        perception_state_channel, 
        map, 
        target_road_segment_id, 
        socket,  gt_channel)
    goal = map[target_road_segment_id]
    initial_localization_state = take!(gt_channel)

    ego_position = initial_localization_state.position
    @info "initial position: $ego_position"
    for i in map.path
        if if_in_segments(i, ego_position)
            initial_segment = i
            @info "intial segment: $initial_segment"
        end
    end
    
    res = a_star_solver(map, initial_segment, goal)
    for i in 1:200
        latest_localization_state = take!(gt_channel)
        # @info "latest_localization_state: $latest_localization_state"
        # latest_perception_state = fetch(perception_state_channel)
        if if_in_segments(goal, latest_localization_state.position)
            break
        end
        for i in res.path
            if if_in_segments(i,latest_localization_state.position)
                cur_seg = i
            end
        end        
        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        curvature = cur_seg.lane_boundaries[1].curvature
        if !isapprox(curvature, 0.0; atol=1e-6)
            steering_angle = curvature
        end
        target_vel = cur_seg.speed_limit
        cmd = VehicleCommand(steering_angle, 10, true)
        serialize(socket, cmd)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = VehicleSim.training_map()
    
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    #localization_state_channel = Channel{MyLocalizationType}(1)
    #perception_state_channel = Channel{MyPerceptionType}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id
        for meas in measurement_msg.measurements
            if meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, socket)
end

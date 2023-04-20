include("./tools/astar.jl")
#include("./trajectory/HW2.jl")

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
    orientation::SVector{4, Float64}
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

    @info "localize function"
    μ =[0, 0, 2.645, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]


    Σ=Diagonal([50,50,0.5,1, 1, 1, 1, 3, 3, 3, 0.5, 0.5, 0.5])


    initial_gps = GPSMeasurement( 0.0, 0.0, 0.0, 0.05)
    initial_imu = IMUMeasurement( 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
  
    fresh_gps_meas = [initial_gps,] # Just to set the initial time to be 0
    fresh_imu_meas = [initial_imu,]
    latest_meas_time = -Inf
    # all_meas = []
    # push!(all_meas, initial_imu)
    # push!(all_meas, initial_gps)

    first_imu = true
    first_gps = true

    while true
        sleep(0.001)
        #@info "in loop"
        all_meas = []

        # time::Float64
        # lat::Float64
        # long::Float64
        # heading::Float64
        while isready(gps_channel)
            sleep(0.001)
            @info "in gps"
            meas = take!(gps_channel)
            if(first_gps)

                μ[1] = meas.lat
                μ[2] = meas.long
                quat = euler_to_quaternion(meas.heading, 0, 0)
                μ[10] = quat[4]
                μ[11] = quat[1]
                μ[12] = quat[2]
                μ[13] = quat[3]
                first_gps = false
            end                
            @info "gpsmeas : $(meas)"
            push!(all_meas, meas)

        end

        while isready(imu_channel)
            sleep(0.001)
            @info "in imu"
            meas = take!(imu_channel)
            if(first_imu)
                μ[4] = meas.linear_vel[1]
                μ[5] = meas.linear_vel[2]
                μ[6] = meas.linear_vel[3]
                μ[7] = meas.angular_vel[1]
                μ[8] = meas.angular_vel[2]
                μ[9] = meas.angular_vel[3]
                first_imu = false
            end
            @info "imumeas : $(meas)"
            push!(all_meas, meas)
        end

       # @info "all_meas : $(all_meas)"
        sorted_all_meas = sort(all_meas, by = x -> x.time)

        start = 1

        for i in 1 : length(sorted_all_meas)
            if sorted_all_meas[i].time >= latest_meas_time
                start = i 
                break
            end
        end
       
        # throw away any measuremetns in all_meas that are from BEFORE last_meas_time

        if length(all_meas) == 0 
            continue
        else
            for i in 1 : length(sorted_all_meas)
            

                dt = sorted_all_meas[i].time - latest_meas_time
                if dt == Inf 
                    dt = 0.05
                end
            @info "211"
            # @info "$(sorted_all_meas[i])"
            # @info "$(dt)"
            # @info "$(μ)"
            # @info "$(Σ)"
            # @info "$(localization_state_channel)"
          #      @info "sorted : $(sorted_all_meas[i])"
                filter(sorted_all_meas[i], dt, μ , Σ,  localization_state_channel)
                latest_meas_time = sorted_all_meas[i].time
            end
        end
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
        #@info "center: $center, r^2: $r"
        if rad_1 < rad_2
            min = rad_1#min(rad_1,rad_2)
            max = rad_2#max(rad_1,rad_2)
        else
            min = rad_2#min(rad_1,rad_2)
            max = rad_1#max(rad_1,rad_2)
        end
        if r < min*min
            return false
        end
        if r > max*max
            return false
        end
        if r > min*min
            if max*max  < r
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

function get_lane_half_space(lane_boundary::LaneBoundary, lane_width::Float64)
    pt_b = lane_boundary.pt_b
    pt_a = lane_boundary.pt_a 
    line_direction = pt_b - pt_a
    line_normal = SVector(-line_direction[2], line_direction[1])
    
    # Normalize the normal vector
    line_normal /= norm(line_normal)
    
    # Compute the distance from the line segment to the origin
    line_distance = dot(line_normal, pt_a)
    
    # Define the half space
    half_space_normal = line_normal
    half_space_distance = line_distance + lane_width/2
    
    return HalfSpace(half_space_normal, half_space_distance)
end

function decision_making(gt_channel ,map, target_channel, ego_vehicle_id_channel, socket)
    gt_vehicle_states = []
    current_segment = map[32]
    current_position = [0.0, 0.0]
    target_road_segment_id = 101
    ego_vehicle_id = 1

    # trace the segments that the car has been through
    path = RoadSegment[]

    while true
        @info "begining decision_making"
        #latest_localization_state = fetch(localization_state_channel)
        #latest_perception_state = fetch(perception_state_channel)

        if isready(target_channel)
            target_road_segment_id = fetch(target_channel)
        end

        if isready(ego_vehicle_id_channel)
            ego_vehicle_id = fetch(ego_vehicle_id_channel)
        end

        @info "current ego vehicle id: $ego_vehicle_id"

        while isready(gt_channel)
            meas = take!(gt_channel)
            if meas.vehicle_id == ego_vehicle_id
                gt_vehicle_states = meas
                @info "updated"
            end
        end
        #sleep(1)

        @info gt_vehicle_states

        if gt_vehicle_states != []
            current_position = gt_vehicle_states.position[1:2]
        end

        @info "searching current segment"
        @info current_position

        # search all map_segments
        for (key,value) in map
            if if_in_segments(map[key], current_position)
                current_segment = map[key]
                @info "current segment: $current_segment"
            end
        end

        @info "found segment"
        @info "current segment"
        @info current_segment
        @info "target segment"
        @info map[target_road_segment_id]

        # path finding A_star
        res = a_star_solver(map, current_segment, map[target_road_segment_id])
        #@info res
        ####### print out the whole path from start point to end point ######
        for i in res.path
            print(i.id)
            println(i.children)
        end

        @info "in the decision_making"
        steering_angle = 0.0
        target_vel = 3.5#current_segment.speed_limit
        lb_1 = current_segment.lane_boundaries[1]
        lb_2 = current_segment.lane_boundaries[2]
        #if isapprox(lb_1.curvature, 0.0; atol=1e-6)
        half_space_1 = get_lane_half_space(lb_1,10.0)
        half_space_2 = get_lane_half_space(lb_2,10.0)

        #pos = (half_space.a)'*current_position
        #if !(current_segment in path)
            #push!(path, current_segment)

            
        
        if (half_space_1.a)'*(current_position-half_space_1.a*5.7/2) - half_space_1.b > 3
            @info "steering angle left"
            steering_angle = 0.2
        elseif (half_space_2.a)'*(current_position-half_space_2.a*5.7/2) - half_space_2.b > 3
            @info "steering angle right"
            steering_angle = -0.2
        end
        
            #end
        #end
        if !isapprox(lb_1.curvature, 0.0; atol=1e-6)
            if !(current_segment in path)
                push!(path, current_segment)
                sleep(2)
                target_vel = 1.5
                if abs(lb_1.curvature) > abs(lb_2.curvature)
                    steering_angle = 1.5708
                else 
                    steering_angle = -1.5708
                end
            else
                sleep(3)
            end
        end

        count = 0
        if current_segment.lane_types == "stop_sign"
            target_vel = 0.0
            count += 1
            #sleep(3)
        end
        if count == 1
            sleep(3)
        end
        
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
        @info "end decision_making"
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

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)
    target_channel = Channel(1)
    ego_vehicle_id_channel = Channel(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while true
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        #measurement_msg = deserialize(socket)

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
        if !isfull(target_channel)
            put!(target_channel, target_map_segment)
        end
        
        ego_vehicle_id = measurement_msg.vehicle_id
        if !isfull(ego_vehicle_id_channel)
            put!(ego_vehicle_id_channel, ego_vehicle_id)
        end

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
    #@async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(gt_channel ,map_segments, target_channel, ego_vehicle_id_channel, socket)
end

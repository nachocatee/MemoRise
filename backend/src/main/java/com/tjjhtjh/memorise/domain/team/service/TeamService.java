package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.team.service.dto.request.CreateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.InviteMemberRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.KickMemberRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.UpdateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.response.CreateTeamResponse;
import com.tjjhtjh.memorise.domain.team.service.dto.response.InviteMemberResponse;
import com.tjjhtjh.memorise.domain.team.service.dto.response.TeamDetailResponse;

public interface TeamService {
    CreateTeamResponse createTeam(CreateTeamRequest createTeamRequest);
    TeamDetailResponse getTeamDetailInfo(Long teamSeq, Long userSeq);
    InviteMemberResponse inviteMember(Long teamSeq, InviteMemberRequest inviteMemberRequest);
    void kickMember(Long teamSeq, KickMemberRequest kickMemberRequest);
    TeamDetailResponse updateTeam(Long teamSeq, UpdateTeamRequest updateTeamRequest);
}

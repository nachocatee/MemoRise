package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.team.service.dto.request.*;
import com.tjjhtjh.memorise.domain.team.service.dto.response.*;

import java.util.List;

public interface TeamService {
    CreateTeamResponse createTeam(CreateTeamRequest createTeamRequest);
    TeamDetailResponse getTeamDetailInfo(Long teamSeq, Long userSeq);
    List<TeamListResponse> getTeamList(Long userSeq, String keyword);
    InviteMemberResponse inviteMember(Long teamSeq, InviteMemberRequest inviteMemberRequest);
    List<InviteUserListResponse> getInviteUserList(Long teamSeq, Long userSeq, String keyword);
    void kickMember(Long teamSeq, KickMemberRequest kickMemberRequest);
    UpdateTeamResponse updateTeam(Long teamSeq, UpdateTeamRequest updateTeamRequest);
    List<MyTeamListResponse> getMyTeamList(Long userSeq);
    EnterTeamResponse enterTeam(Long teamSeq, EnterTeamRequest enterTeamRequest);
    ExitTeamResponse exitTeam(Long teamSeq, Long userSeq);
}

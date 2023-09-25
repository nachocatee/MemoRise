package com.tjjhtjh.memorise.domain.team.repository;

import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.team.service.dto.response.InviteUserListResponse;
import com.tjjhtjh.memorise.domain.team.service.dto.response.TeamListResponse;

import java.util.List;

public interface TeamSupportRepository {
    List<InviteUserListResponse> findInviteUserList(Long teamSeq, Long userSeq, String keyword);
    List<String> findUserProfiles(Long teamSeq, Long userSeq);
    List<Team> findAllByContainsKeyword(Long userSeq, String keyword);
}

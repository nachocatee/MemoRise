package com.tjjhtjh.memorise.domain.team.repository;

import com.tjjhtjh.memorise.domain.team.service.dto.response.InviteUserListResponse;

import java.util.List;

public interface TeamSupportRepository {
    List<InviteUserListResponse> findInviteUserList(Long teamSeq, Long userSeq, String nickname, String email);
}

package com.tjjhtjh.memorise.domain.team.repository;

import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;

import java.util.List;

public interface TeamUserSupportRepository {
    List<Long> findAllUserByTeamSeq(Long teamSeq);
    TeamUser findByTeamSeqAndUserSeq(Long teamSeq, Long userSeq);
    List<Long> findAllTeamByUserSeq(Long userSeq);
}

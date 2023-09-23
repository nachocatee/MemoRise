package com.tjjhtjh.memorise.domain.team.repository;

import java.util.List;

public interface TeamUserSupportRepository {
    List<Long> findTeamUserSeqByTeamSeq(Long teamSeq);

}

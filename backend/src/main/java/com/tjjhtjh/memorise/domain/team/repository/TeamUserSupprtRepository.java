package com.tjjhtjh.memorise.domain.team.repository;

import java.util.List;

public interface TeamUserSupprtRepository {
    List<Long> findTeamUserSeqByTeamSeq(Long teamSeq);

}

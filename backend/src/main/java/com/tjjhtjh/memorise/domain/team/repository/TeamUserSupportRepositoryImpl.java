package com.tjjhtjh.memorise.domain.team.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

import static com.tjjhtjh.memorise.domain.team.repository.entity.QTeamUser.teamUser;


@Repository
@RequiredArgsConstructor
public class TeamUserSupportRepositoryImpl implements TeamUserSupportRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public List<Long> findAllUserByTeamSeq(Long teamSeq) {
        return jpaQueryFactory
            .select(teamUser.user.userSeq)
            .from(teamUser)
            .where(teamUser.team.teamSeq.eq(teamSeq))
            .fetch();
    }

    @Override
    public TeamUser findByTeamSeqAndUserSeq(Long teamSeq, Long userSeq) {
        return jpaQueryFactory
            .selectFrom(teamUser)
            .where(teamUser.team.teamSeq.eq(teamSeq).and(teamUser.user.userSeq.eq(userSeq)))
            .fetchOne();
    }

    @Override
    public List<Long> findAllTeamByUserSeq(Long userSeq) {
        return jpaQueryFactory
            .select(teamUser.team.teamSeq)
            .from(teamUser)
            .where(teamUser.user.userSeq.eq(userSeq))
            .fetch();
    }
}

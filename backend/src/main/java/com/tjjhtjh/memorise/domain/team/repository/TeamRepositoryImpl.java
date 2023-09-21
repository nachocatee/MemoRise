package com.tjjhtjh.memorise.domain.team.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

public class TeamRepositoryImpl extends QuerydslRepositorySupport implements TeamRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public TeamRepositoryImpl(JPAQueryFactory jpaQueryFactory) {
        super(Team.class);
        this.queryFactory = jpaQueryFactory;
    }


}

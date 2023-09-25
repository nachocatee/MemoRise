package com.tjjhtjh.memorise.domain.team.repository;

import com.querydsl.core.BooleanBuilder;
import com.querydsl.core.types.Projections;
import com.querydsl.jpa.JPAExpressions;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.team.service.dto.response.InviteUserListResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;

import static com.tjjhtjh.memorise.domain.team.repository.entity.QTeam.team;
import static com.tjjhtjh.memorise.domain.team.repository.entity.QTeamUser.teamUser;
import static com.tjjhtjh.memorise.domain.user.repository.entity.QUser.user;

@Repository
@RequiredArgsConstructor
public class TeamSupportRepositoryImpl implements TeamSupportRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public List<InviteUserListResponse> findInviteUserList(Long teamSeq, Long userSeq, String keyword) {
        BooleanBuilder builder = new BooleanBuilder();
        if(keyword != null) {
            builder.and(user.nickname.contains(keyword).or(user.email.contains(keyword)));
        }
        return jpaQueryFactory
                .select(Projections.constructor(InviteUserListResponse.class,
                        user,
                        JPAExpressions
                                .selectFrom(teamUser)
                                .where(teamUser.team.teamSeq.eq(teamSeq).and(teamUser.user.userSeq.eq(user.userSeq)))
                                .exists()))
                .from(user)
                .where(user.isDeleted.eq(0).and(builder))
                .orderBy(user.nickname.asc(), user.email.asc())
                .fetch();
    }

//    @Override
//    public List<TeamListResponse> findTeamList(Long userSeq, String keyword) {
//        BooleanBuilder builder = new BooleanBuilder();
//        if(keyword != null) {
//            builder.and(team.name.contains(keyword));
//        }
//        return jpaQueryFactory
//                .select(Projections.constructor(TeamListResponse.class,
//                        team.teamSeq,
//                        team.name,
//                        team.owner,
//                        user.userSeq,
//                        user.profile,
//                        Expressions.asString(JPAExpressions
//                                .select(user.profile)
//                                .from(user)
//                                .where(user.userSeq.eq(team.owner).and(user.isDeleted.eq(0)))
//                                .fetchOne()),
//                        findTeamMemberProfiles(Expressions.asNumber((Expression<Long>) team.teamSeq), userSeq),
//                        JPAExpressions
//                                .selectFrom(teamUser)
//                                .where(teamUser.team.teamSeq.eq(team.teamSeq).and(teamUser.user.userSeq.eq(userSeq)))
//                                .exists() != null)
//                )
//                .from(team)
//                .where(builder)
//                .orderBy(team.name.asc())
//                .fetch();
//    }

//    public String findTeamMemberProfiles(Long teamSeq, Long userSeq) {
//        return jpaQueryFactory
//                .select(user.profile)
//                .from(user)
//                .where(user.userSeq.in(
//                                JPAExpressions
//                                        .select(user.userSeq)
//                                        .from(teamUser)
//                                        .join(user).on(teamUser.user.userSeq.eq(user.userSeq))
//                                        .where(teamUser.team.teamSeq.eq(team.teamSeq)
//                                                .and(user.userSeq.ne(userSeq))
//                                                .and(user.userSeq.ne(team.owner)))
//                                        .limit(4)
//                                        .fetch()
//                        )
//                )
//                .fetchOne();
//    }
}

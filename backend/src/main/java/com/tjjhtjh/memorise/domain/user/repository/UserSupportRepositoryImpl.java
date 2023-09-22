package com.tjjhtjh.memorise.domain.user.repository;

import com.querydsl.core.BooleanBuilder;
import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import com.querydsl.core.types.dsl.BooleanExpression;


import static com.tjjhtjh.memorise.domain.user.repository.entity.QUser.user;

@Repository
@RequiredArgsConstructor
public class UserSupportRepositoryImpl implements UserSupportRepository {

    private final JPAQueryFactory jpaQueryFactory;

    @Override
    public UserListResponse findByNicknameContainingAndIsDeletedFalse(String nickname) {
        BooleanBuilder builder = new BooleanBuilder();
        if(nickname != null) {
            builder.and(user.nickname.contains(nickname));
        }

        return jpaQueryFactory
            .select(Projections.constructor(UserListResponse.class,
                user.userSeq, user.nickname, user.email))
            .from(user)
            .where(user.isDeleted.eq(0).and(builder))
            .fetchOne();
    }

}

package com.tjjhtjh.memorise.domain.tag.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedUserResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;

import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedUser.taggedUser;

public class TaggedUserRepositoryImpl extends QuerydslRepositorySupport implements TaggedUserRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public TaggedUserRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Memo.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public List<TaggedUser> findByTaggedUserWithMemoSeq(Long memoId) {
        return queryFactory.selectFrom(taggedUser).where(taggedUser.memo.memoSeq.eq(memoId)).fetch();
    }

    @Override
    public List<TaggedUserResponse> findByTaggedUserList(Long memoSeq) {
        return queryFactory.select(Projections.fields(TaggedUserResponse.class,taggedUser.user.nickname,taggedUser.taggedUserSeq, taggedUser.user.userSeq))
                .from(taggedUser).where(taggedUser.memo.memoSeq.eq(memoSeq)).fetch();
    }
}

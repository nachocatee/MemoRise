package com.tjjhtjh.memorise.domain.tag.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
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
    public List<Long> findByTaggedListOfMe(Long userSeq) {
        return queryFactory.select(taggedUser.memo.memoSeq).from(taggedUser).where(taggedUser.user.userSeq.eq(userSeq)).fetch();
    }
}

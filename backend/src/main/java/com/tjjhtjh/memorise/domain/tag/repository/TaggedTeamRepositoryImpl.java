package com.tjjhtjh.memorise.domain.tag.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;

import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedTeam.taggedTeam;

public class TaggedTeamRepositoryImpl extends QuerydslRepositorySupport implements TaggedTeamRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public TaggedTeamRepositoryImpl(JPAQueryFactory jpaQueryFactory) {
        super(TaggedTeam.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public List<TaggedTeam> findByTaggedTeamWithMemoSeq(Long memoId) {
        return queryFactory.selectFrom(taggedTeam).where(taggedTeam.memo.memoSeq.eq(memoId)).fetch();
    }
}

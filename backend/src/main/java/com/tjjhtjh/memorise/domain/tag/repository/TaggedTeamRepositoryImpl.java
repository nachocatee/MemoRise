package com.tjjhtjh.memorise.domain.tag.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedTeam;
import com.tjjhtjh.memorise.domain.tag.service.dto.response.TaggedTeamResponse;
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
    public List<TaggedTeam> findByTaggedTeamWithMemoSeq(Long memoSeq) {
        return queryFactory.selectFrom(taggedTeam).where(taggedTeam.memo.memoSeq.eq(memoSeq)).fetch();
    }

    @Override
    public List<TaggedTeamResponse> findByTaggedTeamList(Long memoSeq) {
        return queryFactory.select(Projections.fields(TaggedTeamResponse.class,taggedTeam.team.name))
                .from(taggedTeam).where(taggedTeam.memo.memoSeq.eq(memoSeq)).fetch();
    }
}

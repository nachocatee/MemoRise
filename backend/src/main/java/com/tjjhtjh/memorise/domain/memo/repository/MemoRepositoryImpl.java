package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.BooleanBuilder;
import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoDetailResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;
import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedUser.taggedUser;

@Repository
public class MemoRepositoryImpl extends QuerydslRepositorySupport implements MemoRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public MemoRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Memo.class);
        this.queryFactory = jpaQueryFactory;
    }
    @Override
    public List<MemoResponse> findWrittenByMeOrOpenMemoOrTaggedMemo(Long itemSeq, Long userSeq) {

        BooleanBuilder builder = new BooleanBuilder();
        builder.and(memo.user.userSeq.eq(userSeq).and(memo.item.itemSeq.eq(itemSeq)))  // 내가 작성했거나
                .or(memo.accessType.eq(AccessType.OPEN).and(memo.item.itemSeq.eq(itemSeq)))  // 공개된 메모거나
                .or(memo.accessType.eq(AccessType.RESTRICT)
                        .and(memo.item.itemSeq.eq(itemSeq)
                        .and(taggedUser.user.userSeq.eq(userSeq)).and(taggedUser.memo.memoSeq.eq(memo.memoSeq))));

        return queryFactory.select(Projections.fields
                (MemoResponse.class,
                        memo.user.nickname.as("nickname"), memo.updatedAt,memo.content,memo.accessType,memo.file))
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(taggedUser).on(memo.memoSeq.eq(taggedUser.memo.memoSeq))
                .where(memo.isDeleted.eq(0).and(builder))
                .groupBy(memo.memoSeq)
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }

    @Override
    public Optional<MemoDetailResponse> detailMemo(Long memoId) {
        return queryFactory.select(Projections.fields(MemoDetailResponse.class,
                memo.user.nickname.as("nickname"), memo.updatedAt,memo.content,memo.file))
                .from(memo)
                .leftJoin(memo.user)
                .where(memo.memoSeq.eq(memoId))
                .stream().findAny();
    }

}
